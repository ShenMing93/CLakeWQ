"""Independent read-only QA for annual and climatological OLCI lake statistics.
No statistics-production functions are imported. Reads source products directly.
Only the JSON QA report is written. Production expects consecutive source years;
--allow-nonconsecutive-years is solely for explicitly labelled pilot checks.
"""
from __future__ import annotations
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import sys
import time
import traceback
import zipfile
sys.dont_write_bytecode=True
import fiona
import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.warp import transform_geom
from affine import Affine
from shapely.geometry import shape
from shapely import normalize
import zarr
zarr.config.set({'async.concurrency':1,'threading.max_workers':1})
BASE=Path(__file__).resolve().parent
TAGS=('Chla','SDD','SPM')
KINDS=('climatology_month','annual','climatology_annual')
FULL_REGIONS={'YGPL','EPL'}
SEASONAL_REGIONS={'NPML','TPL','IMXL'}

def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def sha256(path):
    result=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(4*1024**2),b''):
            result.update(block)
    return result.hexdigest()

def config_path(cfg,key):
    path=Path(cfg[key])
    return path if path.is_absolute() else BASE/path

def scalar(value):
    return value.item() if isinstance(value,np.generic) else value

def same_float(value,expected):
    if expected is None:
        return value is None or (isinstance(value,float) and math.isnan(value))
    if value is None:
        return False
    return math.isclose(float(value),float(expected),rel_tol=5e-10,abs_tol=5e-10)

def all_values(obj):
    if isinstance(obj,dict):
        return [v for x in obj.values() for v in all_values(x)]
    if isinstance(obj,list):
        return [v for x in obj for v in all_values(x)]
    return [obj]

class Verifier:
    def __init__(self,args):
        self.args=args;self.cfg=read_json(args.config)
        self.output=Path(args.output_root) if args.output_root else config_path(self.cfg,'output_root')
        self.source=config_path(self.cfg,'source_root')
        self.boundary_path=config_path(self.cfg,'boundaries')
        self.features=read_json(self.boundary_path)['features']
        configured=self.cfg.get('lake_ids')
        if configured is not None and (not isinstance(configured,list) or not configured):
            raise ValueError('Config lake_ids must be a nonempty list')
        aliases={v for f in self.features for v in (f['properties']['lake_id'],f['properties']['code'],f['properties']['name'])}
        if configured is not None and set(configured)-aliases:
            raise ValueError('Unknown configured lakes: '+str(sorted(set(configured)-aliases)))
        configured_features=self.features if configured is None else [f for f in self.features if set(configured).intersection({f['properties']['lake_id'],f['properties']['code'],f['properties']['name']})]
        self.configured_lake_ids={f['properties']['lake_id'] for f in configured_features}
        selectors=args.lake if args.lake else configured
        self.subset_requested=selectors is not None
        if selectors is not None:
            wanted=set(selectors)
            aliases={v for f in self.features for v in (f['properties']['lake_id'],f['properties']['code'],f['properties']['name'])}
            if wanted-aliases:
                raise ValueError('Unknown requested lakes: '+str(sorted(wanted-aliases)))
            self.features=[f for f in self.features if wanted.intersection({f['properties']['lake_id'],f['properties']['code'],f['properties']['name']})]
            if not self.features:
                raise ValueError('No requested lakes in boundary snapshot')
        self.selected_lake_ids={f['properties']['lake_id'] for f in self.features}
        if not self.selected_lake_ids<=self.configured_lake_ids:
            raise ValueError('CLI lakes must belong to the configured lake scope')
        self.full_config_selected=self.selected_lake_ids==self.configured_lake_ids
        self.years=[int(y) for y in self.cfg['years']]
        self.coverage=float(self.cfg['coverage_fraction'])
        self.report={'status':'running','started_local':time.strftime('%Y-%m-%dT%H:%M:%S'),'config':str(Path(args.config).resolve()),'output_root':str(self.output),'years':self.years,'coverage_fraction':self.coverage,'pilot_allow_nonconsecutive_years':args.allow_nonconsecutive_years,'source_sampling_enabled':not args.skip_source_samples,'checks_passed':0,'geometry':[],'maps':[],'source_samples':[],'failures':[]}
        self.expected_rows={};self.sample_data={};self.source_meta={};self.original_stamps={}
        self.report_path=Path(args.report)
        self.check(self.years==sorted(set(self.years)),'Years must be sorted and unique')
        if not args.allow_nonconsecutive_years:
            self.check(self.years==list(range(min(self.years),max(self.years)+1)),'Production climatology needs consecutive years')
        self.check(self.coverage==0.05,'Requested coverage fraction is 5%')
        self.check(len({f['properties']['lake_id'] for f in self.features})==len(self.features),'Duplicate lake IDs')
        default_samples=['H53A302_Xingyun','P65B201_Sailimu','A22B201_Chagan']
        selected={f['properties']['lake_id']:f for f in self.features}
        requested=args.sample_lakes or (list(selected) if self.subset_requested else default_samples)
        self.check(not(set(requested)-set(selected)),'Sample lake IDs must belong to selected lakes')
        self.sample_lakes=list(dict.fromkeys(requested))
        if not args.skip_source_samples and not self.subset_requested:
            self.check(len({selected[x]['properties']['region'] for x in self.sample_lakes})>=3,'Independent source samples need at least three different lake regions')
        self.report['selected_lakes']=list(selected)
        self.report['source_sample_lakes']=self.sample_lakes
        self.report['subset_requested']=self.subset_requested
        self.report['configured_lakes']=sorted(self.configured_lake_ids)
        self.report['full_config_selected']=self.full_config_selected

    def check(self,condition,message):
        if not bool(condition):
            raise AssertionError(message)
        self.report['checks_passed']+=1

    def verify_manifest_lake_scope(self,manifest):
        keysets=[]
        for name in ('lake_fingerprints','input_fingerprints'):
            mapping=manifest.get(name)
            self.check(isinstance(mapping,dict),'Missing manifest lake mapping '+name)
            keys=set(mapping)
            if self.full_config_selected:
                self.check(keys==self.configured_lake_ids,'Manifest lake keys differ from complete configuration: '+name)
            else:
                self.check(self.selected_lake_ids<=keys<=self.configured_lake_ids,'Manifest lake keys exceed configuration or omit selected lakes: '+name)
            keysets.append(keys)
        self.check(keysets[0]==keysets[1],'Manifest fingerprint lake mappings disagree')

    def save(self):
        self.report_path.parent.mkdir(parents=True,exist_ok=True)
        self.report_path.write_text(json.dumps(self.report,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')

    def original_stamp(self,path):
        p=Path(path);s=p.stat();self.original_stamps[str(p)]=(s.st_size,s.st_mtime_ns)

    def month_selection(self,region):
        self.check(region in FULL_REGIONS|SEASONAL_REGIONS,'Unknown lake region '+region)
        return np.array([region in FULL_REGIONS or 5<=m<=10 for m in range(1,13)],dtype=bool)

    def verify_boundary_snapshot(self):
        provenance_path=self.boundary_path.with_name('boundaries_provenance.json')
        provenance=read_json(provenance_path)
        source_path=Path(provenance['source_shapefile'])
        self.original_stamp(source_path)
        wanted={f['properties']['code']:f for f in self.features};original={}
        with fiona.open(source_path) as collection:
            self.check(rasterio.crs.CRS.from_user_input(collection.crs)==rasterio.crs.CRS.from_epsg(4326),'Source SHP CRS must match snapshot EPSG4326')
            for f in collection:
                code=f['properties']['lakeID']
                if code in wanted:
                    self.check(code not in original,'Duplicate source SHP lakeID '+code)
                    original[code]=shape(f['geometry'])
        self.original_geometry={}
        for feature in self.features:
            p=feature['properties'];a=original[p['code']];b=shape(feature['geometry'])
            self.check(normalize(a).equals_exact(normalize(b),0),'Boundary coordinates differ from original SHP: '+p['lake_id'])
            self.original_geometry[p['lake_id']]=a
            self.report['geometry'].append({'lake':p['lake_id'],'source_geometry_type':a.geom_type,'normalized_coordinates_exact':True,'source_valid':bool(a.is_valid)})
        self.report['source_boundary']={'path':str(source_path),'snapshot_sha256':sha256(self.boundary_path),'source_shp_sha256':sha256(source_path)}

    def read_source_metadata(self,feature):
        p=feature['properties'];lake=p['lake_id'];used=self.month_selection(p['region']);metadata={}
        expected_grid=None
        for year in self.years:
            path=self.source/lake/f'{year}.zarr.zip';self.original_stamp(path)
            with zarr.storage.ZipStore(str(path),mode='r') as store:
                g=zarr.open_group(store=store,mode='r');attrs=dict(g.attrs)
                times=g['acquisition_time'][:]
                months=np.array([int(str(int(t))[4:6]) for t in times],dtype=np.int8)
                self.check(all(str(int(t))[:4]==str(year) for t in times),'Source time year differs')
                grid=(g['Chla'].shape[-2:],tuple(attrs['transform'][:6]),attrs['crs_wkt'])
                self.check(g['Chla'].shape[0]==len(times),'Source time count differs')
                if expected_grid is None:expected_grid=grid
                self.check(grid==expected_grid,'Source grids differ across years '+lake)
                for tag in TAGS:self.check(g[tag].shape==g['Chla'].shape,'Source product shapes differ')
                metadata[year]={'path':str(path),'times':times,'months':months,'eligible':used[months-1],'attrs':attrs,'shape':g['Chla'].shape,'run_fingerprint':attrs['run_fingerprint']}
        self.source_meta[lake]=metadata
        return metadata,expected_grid,used

    def validate_zip(self,path):
        self.original_stamp(path)
        with zipfile.ZipFile(path) as archive:
            names=archive.namelist()
            self.check(len(names)==len(set(names)),'Duplicate ZIP member '+str(path))
            self.check(archive.testzip() is None,'ZIP CRC failure '+str(path))

    def coordinates(self,counts,polygon):
        flat=counts.reshape(-1);inside=np.flatnonzero(polygon.ravel());observed=np.flatnonzero(flat>0);zero=np.flatnonzero((flat==0)&polygon.ravel());outside=np.flatnonzero(~polygon.ravel())
        points={0,len(flat)-1,len(flat)//2}
        for candidates,number in ((observed,20),(zero,4),(outside,4),(inside,4)):
            if len(candidates):points.update(candidates[np.linspace(0,len(candidates)-1,min(number,len(candidates)),dtype=int)].tolist())
        if len(observed):
            order=observed[np.argsort(flat[observed])]
            points.update(order[:3].tolist());points.update(order[-3:].tolist())
        return np.unravel_index(np.array(sorted(points)),counts.shape)

    def verify_lake_maps(self,feature):
        p=feature['properties'];lake=p['lake_id']
        sources,grid,used=self.read_source_metadata(feature)
        hw,affine_values,crs_wkt=grid;affine=Affine(*affine_values)
        geom=feature['geometry'];crs=rasterio.crs.CRS.from_wkt(crs_wkt)
        if crs!=rasterio.crs.CRS.from_epsg(4326):geom=transform_geom('EPSG:4326',crs,geom)
        polygon=rasterize([(geom,1)],out_shape=hw,transform=affine,fill=0,all_touched=False,dtype='uint8').astype(bool)
        lakepix=int(polygon.sum());self.check(lakepix>0,'Empty SHP rasterization '+lake)
        annual_n=np.array([int(sources[y]['eligible'].sum()) for y in self.years],dtype=np.uint32)
        monthly_n=np.array([sum(int(np.count_nonzero(s['eligible']&(s['months']==m))) for s in sources.values()) for m in range(1,13)],dtype=np.uint32)
        global_n=int(annual_n.sum());self.check(global_n==int(monthly_n.sum()),'Source selected-date totals differ')
        all_path=self.output/lake/'climatology_annual.zarr.zip'
        with zarr.storage.ZipStore(str(all_path),mode='r') as store:
            g=zarr.open_group(store=store,mode='r')
            global_counts={tag:g[tag+'_nobs'][:] for tag in TAGS}
        sample=None
        if lake in self.sample_lakes and not self.args.skip_source_samples:
            coords=self.coordinates(global_counts['Chla'],polygon)
            sample={'coords':coords,'polygon':polygon[coords],'periods':{},'lake':lake,'region':p['region']}
            self.sample_data[lake]=sample
        annual_count_sum={tag:np.zeros(hw,dtype=np.uint64) for tag in TAGS}
        month_count_sum={tag:np.zeros(hw,dtype=np.uint64) for tag in TAGS}
        for kind in KINDS:
            path=self.output/lake/(kind+'.zarr.zip');self.validate_zip(path)
            with zarr.storage.ZipStore(str(path),mode='r') as store:
                g=zarr.open_group(store=store,mode='r');a=dict(g.attrs)
                self.check(a.get('complete') is True,'Incomplete statistics ZIP '+str(path))
                self.check(list(a.get('sourceyears',a.get('source_years',[])))==self.years,'Source years differ '+str(path))
                self.check(float(a['coverage_fraction'])==self.coverage,'Coverage differs '+str(path))
                self.check(int(a['global_n_scenes'])==global_n,'Global denominator differs '+str(path))
                self.check(tuple(a['transform'][:6])==affine_values,'Statistics transform differs')
                self.check(rasterio.crs.CRS.from_wkt(a['crs_wkt'])==crs,'Statistics CRS differs')
                self.check(a['boundary_sha256']==sha256(self.boundary_path),'Boundary fingerprint differs')
                np.testing.assert_array_equal(g['x'][:],affine.c+(np.arange(hw[1])+.5)*affine.a)
                np.testing.assert_array_equal(g['y'][:],affine.f+(np.arange(hw[0])+.5)*affine.e)
                self.check(g['x'].dtype==np.dtype('float64') and g['y'].dtype==np.dtype('float64'),'Coordinate dtype differs')
                np.testing.assert_array_equal(g['month'][:],np.arange(1,13))
                np.testing.assert_array_equal(g['month_used'][:].astype(bool),used)
                fingerprints=a['input_fingerprints']
                self.check(len(fingerprints)==len(self.years),'Source fingerprint count differs')
                for record in fingerprints:
                    sy=int(record['year']);source=Path(sources[sy]['path']);st=source.stat()
                    self.check(Path(record['path']).resolve()==source.resolve(),'Source fingerprint path differs')
                    self.check(record['size_bytes']==st.st_size and record['mtime_ns']==st.st_mtime_ns,'Source changed after statistics')
                    self.check(isinstance(record['sha256'],str) and len(record['sha256'])==64,'Source SHA256 missing')
                if kind=='climatology_month':
                    shape_expected=(12,*hw);periods=list(range(1,13));nscenes=monthly_n
                elif kind=='annual':
                    np.testing.assert_array_equal(g['year'][:],self.years)
                    shape_expected=(len(self.years),*hw);periods=self.years;nscenes=annual_n
                else:
                    shape_expected=hw;periods=[0];nscenes=np.array([global_n],dtype=np.uint32)
                ns=g['n_scenes'][...]
                self.check(np.asarray(ns).shape==(() if kind=='climatology_annual' else (len(periods),)),'n_scenes dimensions differ')
                np.testing.assert_array_equal(np.atleast_1d(ns),nscenes)
                self.check(g['n_scenes'].dtype==np.dtype('uint32'),'n_scenes must be uint32')
                map_info={'lake':lake,'region':p['region'],'kind':kind,'path':str(path),'shape':list(shape_expected),'n_scenes':[int(n) for n in nscenes],'global_n_scenes':global_n,'zip_crc':'passed','sha256':sha256(path),'periods':[]}
                records={period:{'lake_id':lake,'name_cn':p['name'],'region':p['region'],'year':period if kind=='annual' else None,'month':period if kind=='climatology_month' else None,'yr_start':min(self.years),'yr_end':max(self.years),'nsce':int(nscenes[i]),'n_lakepix':lakepix} for i,period in enumerate(periods)}
                for tag in TAGS:
                    self.check(g[tag].shape==shape_expected and g[tag+'_nobs'].shape==shape_expected,'Map/count shape differs '+tag)
                    self.check(g[tag].dtype==np.dtype('float32') and g[tag+'_nobs'].dtype==np.dtype('uint32'),'Map/count dtype differs '+tag)
                    values=g[tag][...];counts=g[tag+'_nobs'][...]
                    if kind=='annual':annual_count_sum[tag]+=counts.sum(axis=0,dtype=np.uint64)
                    if kind=='climatology_month':month_count_sum[tag]+=counts.sum(axis=0,dtype=np.uint64)
                    for i,period in enumerate(periods):
                        mean=values if kind=='climatology_annual' else values[i]
                        count=counts if kind=='climatology_annual' else counts[i]
                        n=int(nscenes[i]);gc=global_counts[tag]
                        expected_support=polygon&(count>0)&(count>=self.coverage*n)&(gc>=self.coverage*global_n)
                        if n==0 or global_n==0:expected_support[:]=False
                        self.check(not np.isinf(mean).any(),'Infinity in statistics map')
                        np.testing.assert_array_equal(np.isfinite(mean),expected_support)
                        self.check(np.all(count<=n),'nobs exceeds period records')
                        self.check(not count[~polygon].any(),'Counts outside lake SHP')
                        self.check(np.all(mean[np.isfinite(mean)]>0),'Unexpected zero/negative finite mean')
                        if kind=='climatology_month' and not used[period-1]:
                            self.check(n==0 and not count.any() and np.isnan(mean).all(),'Excluded month contains data')
                        finite=mean[np.isfinite(mean)];npix=int(finite.size)
                        records[period][tag]=float(finite.mean(dtype=np.float64)) if npix else None
                        records[period][tag+'_npix']=npix
                        records[period][tag+'_frac']=npix/lakepix
                        if sample is not None:
                            key=(kind,period)
                            entry=sample['periods'].setdefault(key,{'n_scenes':n,'products':{}})
                            entry['products'][tag]={'mean':mean[sample['coords']].copy(),'count':count[sample['coords']].copy()}
                for period,record in records.items():
                    if kind!='climatology_month' or used[period-1]:
                        self.expected_rows[(kind,lake,record['year'],record['month'])]=record
                    map_info['periods'].append(record)
                self.report['maps'].append(map_info)
        for tag in TAGS:
            np.testing.assert_array_equal(annual_count_sum[tag],global_counts[tag])
            np.testing.assert_array_equal(month_count_sum[tag],global_counts[tag])
        self.check(True,'All maps and count pooling passed '+lake)
        self.save()
        print(json.dumps({'verified_lake_maps':lake,'global_n_scenes':global_n,'sample_pixels':len(sample['coords'][0]) if sample else 0}),flush=True)

    def verify_shapefiles(self):
        self.report['shapefiles']=[]
        chosen={f['properties']['lake_id'] for f in self.features}
        for kind in KINDS:
            path=self.output/(kind+'.shp');seen=set();rows_checked=0
            with fiona.open(path) as collection:
                self.check(rasterio.crs.CRS.from_user_input(collection.crs)==rasterio.crs.CRS.from_epsg(4326),'SHP CRS differs')
                for feature in collection:
                    props=dict(feature['properties']);lake=props['lake_id']
                    if lake not in chosen:
                        self.check(not self.full_config_selected and lake in self.configured_lake_ids,'Unexpected lake in output SHP '+str(lake));continue
                    key=(kind,lake,props['year'],props['month'])
                    self.check(key in self.expected_rows,'Unexpected/seasonally excluded SHP period '+str(key))
                    self.check(key not in seen,'Duplicate SHP lake-period '+str(key));seen.add(key)
                    expected=self.expected_rows[key]
                    geom=shape(feature['geometry'])
                    self.check(normalize(geom).equals_exact(normalize(self.original_geometry[lake]),0),'Output SHP geometry altered '+lake)
                    for name in ('lake_id','name_cn','region','year','month','yr_start','yr_end','nsce','n_lakepix'):
                        self.check(props[name]==expected[name],'SHP identity/count differs '+str(key)+' '+name)
                    for tag in TAGS:
                        self.check(same_float(props[tag],expected[tag]),'SHP value differs from saved float32 map mean '+str(key)+' '+tag)
                        self.check(props[tag+'_npix']==expected[tag+'_npix'],'SHP valid pixel count differs')
                        self.check(same_float(props[tag+'_frac'],expected[tag+'_frac']),'SHP spatial coverage differs')
                    rows_checked+=1
            expected_keys={k for k in self.expected_rows if k[0]==kind}
            self.check(seen==expected_keys,'Missing SHP lake-period rows '+kind)
            self.report['shapefiles'].append({'path':str(path),'rows_checked':rows_checked,'geometry_coordinates_exact':True,'all_values_match_saved_float32_map_spatial_mean':True,'null_means_not_zero':True})

    def verify_source_samples(self):
        for lake,sample in self.sample_data.items():
            meta=self.source_meta[lake];ys,xs=sample['coords'];blocks={tag:[] for tag in TAGS};years=[];months=[]
            elapsed=time.perf_counter();excluded_records=0;input_hashes=[]
            with zarr.storage.ZipStore(str(self.output/lake/'climatology_annual.zarr.zip'),mode='r') as store:
                expected_fingerprints={int(r['year']):r for r in zarr.open_group(store=store,mode='r').attrs['input_fingerprints']}
            for year in self.years:
                item=meta[year];eligible=item['eligible'];n=len(eligible);excluded_records+=int((~eligible).sum())
                actual_digest=sha256(item['path'])
                self.check(actual_digest==expected_fingerprints[year]['sha256'],'Sample source SHA256 differs '+lake+' '+str(year))
                input_hashes.append({'year':year,'path':item['path'],'sha256':actual_digest})
                with zarr.storage.ZipStore(item['path'],mode='r') as store:
                    g=zarr.open_group(store=store,mode='r')
                    for start in range(0,n,16):
                        end=min(start+16,n);keep=eligible[start:end]
                        if not keep.any():continue
                        for tag in TAGS:
                            # Product source chunks contain the whole spatial grid.
                            # Decode each time chunk once, then retain selected pixels only.
                            a=g[tag][start:end]
                            blocks[tag].append(a[keep][:,ys,xs])
                        years.extend([year]*int(keep.sum()))
                        months.extend(item['months'][start:end][keep].tolist())
                        if self.args.read_delay>0:time.sleep(self.args.read_delay)
            years=np.asarray(years);months=np.asarray(months);n_global=len(years)
            values={tag:np.concatenate(parts,axis=0) for tag,parts in blocks.items()}
            entry={'lake':lake,'region':sample['region'],'sample_pixel_count':len(ys),'rows':ys.tolist(),'columns':xs.tolist(),'selected_records_including_empty':n_global,'excluded_month_records':excluded_records,'source_sha256_checked':input_hashes,'periods_checked':0,'product_pixel_comparisons':0,'maximum_absolute_difference':0.0,'all_counts_exact':True,'all_float32_means_bitwise_equal':True}
            global_counts={tag:np.isfinite(v).sum(axis=0,dtype=np.uint32) for tag,v in values.items()}
            for (kind,period),stored in sample['periods'].items():
                included=(months==period) if kind=='climatology_month' else ((years==period) if kind=='annual' else np.ones(n_global,dtype=bool))
                n=int(included.sum());self.check(n==stored['n_scenes'],'Independent source period denominator differs')
                for tag in TAGS:
                    a=values[tag][included];finite=np.isfinite(a)
                    count=finite.sum(axis=0,dtype=np.uint32)
                    total=np.where(finite,a,0).sum(axis=0,dtype=np.float64)
                    expected=np.full(len(ys),np.nan,dtype=np.float64)
                    allowed=sample['polygon']&(count>0)&(count>=self.coverage*n)&(global_counts[tag]>=self.coverage*n_global)
                    if n==0:allowed[:]=False
                    np.divide(total,count,out=expected,where=allowed)
                    expected=expected.astype(np.float32)
                    target=stored['products'][tag]
                    np.testing.assert_array_equal(target['count'],count)
                    np.testing.assert_array_equal(target['mean'],expected)
                    supported=np.isfinite(expected)
                    self.check(np.array_equal(target['mean'][supported].view(np.uint32),expected[supported].view(np.uint32)),'Independent float32 map values differ')
                    entry['product_pixel_comparisons']+=len(ys)
                entry['periods_checked']+=1
            entry['seconds']=round(time.perf_counter()-elapsed,3)
            self.report['source_samples'].append(entry);self.save()
            print(json.dumps({'source_recalculation_passed':lake,'pixels':len(ys),'periods':entry['periods_checked'],'records':n_global,'seconds':entry['seconds']}),flush=True)

    def verify_manifest(self):
        manifest_path=self.output/'run_manifest.json';manifest=read_json(manifest_path)
        self.check(manifest.get('complete') is True,'Statistics run manifest is not complete')
        self.verify_manifest_lake_scope(manifest)
        self.check(list(manifest.get('sourceyears',manifest.get('source_years',[])))==self.years,'Run manifest years differ')
        checked=0
        for name,record in manifest['files'].items():
            path=self.output/name
            self.check(path.is_file(),'Manifest file missing '+name)
            self.check(path.stat().st_size==record['size_bytes'],'Manifest file size differs '+name)
            self.check(sha256(path)==record['sha256'],'Manifest file SHA256 differs '+name)
            checked+=1
        for feature in self.features:
            lake=feature['properties']['lake_id']
            records=manifest['input_fingerprints'][lake]
            self.check({int(r['year']) for r in records}==set(self.years),'Run source fingerprint years differ')
            for record in records:
                source=self.source_meta[lake][int(record['year'])]
                self.check(Path(record['path']).resolve()==Path(source['path']).resolve(),'Run source fingerprint path differs')
                stat=Path(source['path']).stat()
                self.check(record['size_bytes']==stat.st_size and record['mtime_ns']==stat.st_mtime_ns,'Run source fingerprint changed')
        self.report['manifest']={'path':str(manifest_path),'sha256':sha256(manifest_path),'files_checked':checked,'complete':True,'run_fingerprint':manifest.get('run_fingerprint')}

    def run(self):
        self.verify_boundary_snapshot()
        for feature in self.features:self.verify_lake_maps(feature)
        self.verify_shapefiles()
        self.verify_manifest()
        if not self.args.skip_source_samples:self.verify_source_samples()
        for name,before in self.original_stamps.items():
            stat=Path(name).stat()
            self.check((stat.st_size,stat.st_mtime_ns)==before,'QA source file changed during validation '+name)
        self.report.update(status='passed',finished_local=time.strftime('%Y-%m-%dT%H:%M:%S'),lakes_verified=len(self.features),zarr_archives_verified=len(self.report['maps']),source_and_output_size_mtime_unchanged=True)
        self.report['limitations']=[]
        if self.args.skip_source_samples:self.report['limitations'].append('Independent source numerical recalculation explicitly skipped; this is structural/coverage/geometry QA only.')
        if self.args.allow_nonconsecutive_years:self.report['limitations'].append('Pilot permits nonconsecutive source years; it is not the final continuous-year climatology verification.')
        self.report['limitations'].append('Full numerical source recalculation uses deterministic sampled pixels in the selected lakes; all saved maps are fully read and checked for support, coverage, geometry and SHP means.')
        self.save()
        print(json.dumps({'status':'passed','lakes':len(self.features),'zarr_archives':len(self.report['maps']),'source_sample_lakes':len(self.report['source_samples']),'report':str(self.report_path)},ensure_ascii=True),flush=True)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default=str(BASE/'config/run.json'))
    parser.add_argument('--output-root')
    parser.add_argument('--report',default=str(BASE/'verification_report.json'))
    parser.add_argument('--lake',nargs='+',help='Explicit pilot/subset lake ID, code or Chinese name')
    parser.add_argument('--sample-lakes',nargs='+',help='Full lake folder IDs; defaults to Xingyun/YGPL, Sailimu/IMXL, Chagan/NPML')
    parser.add_argument('--skip-source-samples',action='store_true')
    parser.add_argument('--allow-nonconsecutive-years',action='store_true',help='Only for explicitly labelled pilot checks')
    parser.add_argument('--read-delay',type=float,default=.02,help='Pause between independent source time-chunk reads')
    args=parser.parse_args();qa=None
    try:
        qa=Verifier(args);qa.run();return 0
    except Exception as exc:
        report={'status':'failed','finished_local':time.strftime('%Y-%m-%dT%H:%M:%S'),'failures':[]}
        if qa is not None:report=qa.report;report['status']='failed';report['finished_local']=time.strftime('%Y-%m-%dT%H:%M:%S')
        report['failures'].append({'type':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc()})
        path=Path(args.report);path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
        traceback.print_exc();return 1

if __name__=='__main__':
    raise SystemExit(main())
