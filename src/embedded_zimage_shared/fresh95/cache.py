"""Already-completed native ZImage responses only; never prepare teachers."""
from dataclasses import fields
from pathlib import Path
import math,torch
from particle_sliders.pg09_backends import PreparedResponse,ResponseRow
from particle_sliders.pg09_images import image_condition_features
from . import PROFILE,ownership
from .util import canonical,equal,read_packet,save_packet,source_record,tree

SCHEMA='ZImage-completed-native-PreparedResponses-explicit-fresh95-v1'

def file_record(value):
    if (not isinstance(value,dict)or set(value)!={'path','bytes','sha256'}
            or not isinstance(value['path'],str)or not Path(value['path']).is_absolute()or'..'in Path(value['path']).parts
            or type(value['bytes'])is not int or value['bytes']<1
            or not isinstance(value['sha256'],str)or len(value['sha256'])!=64
            or any(c not in'0123456789abcdef'for c in value['sha256'])):
        raise ValueError('whole original file byte/SHA origin required')

def finite(value):
    if isinstance(value,torch.Tensor):
        if value.requires_grad or value.layout!=torch.strided or value.is_complex()or not torch.isfinite(value).all():
            raise ValueError('complete frozen finite dense tensors required')
    elif isinstance(value,dict):
        if any(not isinstance(k,str)for k in value):raise ValueError('string cache keys required')
        for item in value.values():finite(item)
    elif isinstance(value,(tuple,list)):
        for item in value:finite(item)
    elif value is not None and type(value)not in(str,int,float,bool):raise ValueError('unsupported cache object')
    elif type(value)is float and not math.isfinite(value):raise ValueError('finite cache scalars required')

def validate_prepared(prepared,config):
    from embedded_zimage_shared import backend
    if not isinstance(prepared,(list,tuple))or len(prepared)<2:raise ValueError('at least2 completed original PreparedResponses required')
    shape=backend._shape(config['response_shape'])
    identity=config['model_identity']
    for item in prepared:
        if type(item)is not PreparedResponse or type(item.row)is not ResponseRow:
            raise ValueError('actual original PreparedResponse/ResponseRow required')
        row=item.row
        if row.schedule is not None or row.scale!=1. or row.strength!=1. or row.guidance!=1.:
            raise ValueError('original native raw-velocity request contract differs')
        # Original check_request is CPU/data-only; this does not call a model.
        for condition in(row.neutral,row.positive,row.negative):
            backend.check_request(condition,row.latent,row.timestep,shape,identity)
        for tensor in(item.target,item.neutral):
            if not isinstance(tensor,torch.Tensor)or tuple(tensor.shape)!=shape or tensor.dtype!=torch.float32 or tensor.device.type!='cpu':
                raise ValueError('whole original CPU FP32 completed response fields required')
        for value in(row.neutral,row.positive,row.negative,row.latent,row.timestep,item.target,item.neutral):finite(value)
    return shape

def dump(item):
    return dict(row={f.name:tree(getattr(item.row,f.name),'cpu')for f in fields(ResponseRow)},
        target=tree(item.target,'cpu'),neutral=tree(item.neutral,'cpu'))

def restore(value,device):
    if (not isinstance(value,dict)or set(value)!={'row','target','neutral'}
            or not isinstance(value['row'],dict)or set(value['row'])!={f.name for f in fields(ResponseRow)}):
        raise ValueError('complete original serialized response row required')
    return PreparedResponse(ResponseRow(**tree(value['row'],device)),tree(value['target'],device),tree(value['neutral'],device))

def origins_valid(origins,prepared):
    if not isinstance(origins,list)or len(origins)!=len(prepared):raise ValueError('each completed original response needs an origin')
    for origin in origins:
        if (not isinstance(origin,dict)or origin.get('scope')not in('completed-native','synthetic-software-only')
                or origin.get('complete_prepared_output')is not True
                or not isinstance(origin.get('whole_output_files'),list)or not origin['whole_output_files']
                or not isinstance(origin.get('source_files'),list)or not origin['source_files']):
            raise ValueError('explicit completed whole-output and Source provenance required')
        for record in origin['whole_output_files']+origin['source_files']:file_record(record)
    canonical(origins)

def teacher_identity(config):
    keys=('runtime','model_identity','response_shape')
    return{k:config[k]for k in keys}

def save_prepared_cache(path,*,config,prepared,teacher_origins,cache_id,scope,owner_admission):
    from .api import configuration
    config=configuration(config);ownership.guard();validate_prepared(prepared,config);origins_valid(teacher_origins,prepared)
    if scope not in('completed-native','synthetic-software-only')or any(x['scope']!=scope for x in teacher_origins):
        raise ValueError('native and synthetic completed origins remain distinct')
    if not isinstance(cache_id,str)or not cache_id.strip()or not callable(owner_admission):raise ValueError('cache ID and owner required')
    request=dict(operation='serialize-completed-cache',profile=PROFILE,cache_id=cache_id,scope=scope,
        teacher_identity=teacher_identity(config),teacher_origins=teacher_origins,model_or_teacher_calls=0)
    if owner_admission(request)is not True:raise ValueError('owner did not authenticate original completed data')
    # The caller authenticates original origins; this serializer only binds
    # supplied completed tensors and the exact original pure feature function.
    return save_packet(path,dict(schema=SCHEMA,profile=PROFILE,cache_id=cache_id,scope=scope,
        teacher_identity=teacher_identity(config),original_sources=ownership.sources()['original'],
        serializer_source=source_record(__file__),teacher_origins=teacher_origins,
        rows=[dump(x)for x in prepared],conditions=image_condition_features(prepared).detach().cpu(),
        complete=True,model_or_teacher_calls=0))

def load_cache(pin,*,config):
    packet=read_packet(pin)
    keys={'schema','profile','cache_id','scope','teacher_identity','original_sources','serializer_source','teacher_origins','rows','conditions','complete','model_or_teacher_calls'}
    if (not isinstance(packet,dict)or set(packet)!=keys or packet['schema']!=SCHEMA or packet['profile']!=PROFILE
            or packet['complete']is not True or packet['model_or_teacher_calls']!=0
            or packet['scope']not in('completed-native','synthetic-software-only')
            or packet['teacher_identity']!=teacher_identity(config)
            or packet['original_sources']!=ownership.sources()['original']or packet['serializer_source']!=source_record(__file__)):
        raise ValueError('exact completed native cache identity/Source required; request-only/old caches refused')
    prepared=[restore(x,'cpu')for x in packet['rows']];shape=validate_prepared(prepared,config);origins_valid(packet['teacher_origins'],prepared)
    if any(x['scope']!=packet['scope']for x in packet['teacher_origins']):raise ValueError('cache origin scope differs')
    finite(packet['conditions'])
    if not equal(image_condition_features(prepared),packet['conditions']):raise ValueError('original image condition-feature arithmetic differs')
    scale=float(torch.cat([(x.target-x.neutral).flatten()for x in prepared]).double().square().mean().sqrt())
    if not 0<=scale<float('inf'):raise ValueError('completed target-delta RMS diagnostic must be finite')
    identity={k:packet[k]for k in('schema','profile','cache_id','scope','teacher_identity','original_sources','serializer_source','teacher_origins')}
    identity.update(whole_file=dict(path=pin.path,bytes=pin.bytes,sha256=pin.sha256),row_count=len(prepared),
        response_shape=list(shape),target_delta_RMS_diagnostic=scale,game_response_scale=1.0)
    return prepared,packet['conditions'],identity

def to_device(prepared,device):return[restore(dump(item),device)for item in prepared]
