"""Original runtime and genuine distribution ownership; reversible new attachments."""
from importlib import metadata
from pathlib import Path
import base64,csv,hashlib,io
from .baseline import ORIGINAL_SOURCES
from .util import source_record

def guard():
    from embedded_zimage_shared import backend
    from particlegan import init as pg_init
    from .policy import INIT_SHA
    root=Path(backend.__file__).absolute().parent
    for name,pin in ORIGINAL_SOURCES.items():
        if source_record(root/name)!=pin:raise RuntimeError('original ZImage Source differs:'+name)
    backend._core_gate()  # Exact original core0.13 Git origin, closure, Diffusers, PG/HNDL guards.
    if metadata.version('particlegan')!='0.9.5'or source_record(pg_init.__file__)['sha256']!=INIT_SHA:
        raise RuntimeError('exact public PG095 initializer required')
    owners=[d for d in metadata.distributions()if(d.metadata.get('Name')or'').lower().replace('_','-')=='embedded-zimage-shared-preview']
    if len(owners)!=1 or owners[0].version!='0.1.4':raise RuntimeError('one genuine owning ZImage distribution required')
    owner=owners[0]
    if Path(owner.locate_file('embedded_zimage_shared/backend.py')).absolute()!=Path(backend.__file__).absolute():
        raise RuntimeError('ZImage import shadowed from its owning distribution')
    records={r[0]:r[1:]for r in csv.reader(io.StringIO(owner.read_text('RECORD')or''))}
    for path in [*(root/name for name in ORIGINAL_SOURCES),*sorted(Path(__file__).absolute().parent.glob('*.py'))]:
        raw=path.read_bytes();relative='embedded_zimage_shared/'+str(path.relative_to(root))
        encoded=base64.urlsafe_b64encode(hashlib.sha256(raw).digest()).rstrip(b'=').decode()
        if records.get(relative)!=['sha256='+encoded,str(len(raw))]:
            raise RuntimeError('genuine original/new-profile RECORD differs:'+relative)
    return dict(profile='original-ZImage-core013-direct-Git-closure-and-owning-RECORD',PG='0.9.5',core='0.13.0')

def sources():
    from embedded_zimage_shared import backend,cuda_backend,execution,graphs,learning_rates
    from particle_sliders import pg09,pg095,pg09_backends,pg09_images,lifecycle,fibo_hndl
    modules=(backend,cuda_backend,execution,graphs,learning_rates,pg09,pg095,pg09_backends,pg09_images,lifecycle,fibo_hndl)
    return dict(own={p.name:source_record(p)for p in sorted(Path(__file__).absolute().parent.glob('*.py'))},
        original={m.__name__:source_record(m.__file__)for m in modules})

MARKER='_zimage_explicit_fresh95_attachment'

class Attachment:
    """Own original LoRA hooks, native checkpoint attributes and frozen flags.

    The caller retains its admitted native model. No unload, offload, dtype or
    replacement module is performed here. Failed cleanup retains only pending
    operations; foreign attachment/checkpoint owners are not overwritten.
    """
    def __init__(self,model):
        if hasattr(model,MARKER):raise ValueError('foreign ZImage fresh owner exists')
        self.model=model;self.closed=self.closing=False;self.network=None
        self.pending_hooks=[];self.pending_checkpoint=[];self.checkpoint_before=()
        self.handle_container=None;self.handle_items=();self.hook_owners={};self.pending_container=False
        self.pending_flags=[(p,p.requires_grad)for p in model.parameters()]
        self.pending_modes=[(m,m.training)for m in model.modules()]
        self.flags=list(self.pending_flags);self.modes=list(self.pending_modes)
        self._capture_owned_flags_modes()
        self.pending_marker=True;setattr(model,MARKER,self)
    def attach(self,initialization,config,cuda):
        from embedded_zimage_shared.graphs import ZImageLoRA
        try:self.model.requires_grad_(False).eval()
        finally:self._capture_owned_flags_modes()
        if cuda and config['gradient_checkpointing']:
            self.checkpoint_before=tuple((m,k,k in vars(m),vars(m).get(k))for m in self.model.modules()
                for k in('gradient_checkpointing','_gradient_checkpointing_func'))
            try:self.model.enable_gradient_checkpointing()
            finally:self._capture_owned_checkpoint_state()
        from embedded_zimage_shared import backend,cuda_backend
        scope=cuda_backend.initialization_scope(initialization,'adapter',cuda_backend.execution_policy(config))if cuda else backend.initialization_scope(initialization,'adapter')
        with scope:
            self.network=ZImageLoRA.__new__(ZImageLoRA)
            try:ZImageLoRA.__init__(self.network,self.model)
            finally:self._capture_hooks()
            if cuda:self.network.to(config['device'])
        self.ids=[(n,id(p))for n,p in self.network.named_parameters()]
        return self.network
    def _capture_owned_flags_modes(self):
        self.owned_flags={id(parameter):parameter.requires_grad for parameter,original in self.flags}
        self.owned_modes={id(module):module.training for module,original in self.modes}
    def _capture_owned_checkpoint_state(self):
        # Only the bounded original enable/response-adapter constructors call this.
        # Core rebinds the function after initial native enablement; preserve the
        # original pre-enable state and own the final/partial constructor state.
        self._capture_owned_flags_modes()  # Final/partial bounded constructor state, not user-operation state.
        self.pending_checkpoint=[(m,k,had,old,k in vars(m),vars(m).get(k))for m,k,had,old in self.checkpoint_before
            if (k in vars(m))!=had or vars(m).get(k)is not old]
    def _capture_hooks(self):
        container=getattr(self.network,'_handles',None)
        if container is None:return  # Original partial constructor attached no hook yet.
        if type(container)is not list:raise RuntimeError('original hook-handle list required')
        self.handle_container=container;self.handle_items=tuple(container)
        self.pending_hooks=list(container);self.pending_container=True
        for handle in container:
            hooks=handle.hooks_dict_ref()
            if hooks is None or handle.id not in hooks:raise RuntimeError('owned original hook callback missing')
            extras=[]
            for reference in getattr(handle,'extra_dict_ref',()):
                extra=reference()
                if extra is not None:extras.append((extra,handle.id in extra,extra.get(handle.id)))
            self.hook_owners[id(handle)]=(hooks,hooks[handle.id],extras)
    def _assert_handle_container(self):
        if self.handle_container is None:return
        current=getattr(self.network,'_handles',None)
        if current is not self.handle_container or len(current)!=len(self.handle_items)or any(a is not b for a,b in zip(current,self.handle_items)):
            raise RuntimeError('foreign hook handle container/items retained')
    def _remove_hook(self,handle):
        self._assert_handle_container()
        hooks,callback,extras=self.hook_owners[id(handle)]
        if handle.hooks_dict_ref()is not hooks or hooks.get(handle.id)is not callback:
            raise RuntimeError('foreign hook callback retained')
        if any((handle.id in extra)!=had or extra.get(handle.id)is not value for extra,had,value in extras):
            raise RuntimeError('foreign hook extra-dictionary ownership retained')
        handle.remove()
    def require_active(self):
        if self.closed or self.closing:raise RuntimeError('ZImage attachment closed/cleanup pending')
        if vars(self.model).get(MARKER)is not self:raise RuntimeError('foreign ZImage attachment marker')
        if [(n,id(p))for n,p in self.network.named_parameters()]!=self.ids:raise RuntimeError('original adapter Parameter identity changed')
        if any(p.requires_grad for p in self.model.parameters()):raise RuntimeError('frozen native model ownership changed')
    @staticmethod
    def _run(pending,operation,errors):
        failed=[]
        for value in pending:
            try:operation(value)
            except BaseException as e:failed.append(value);errors.append(e)
        return failed
    def close(self):
        if self.closed:return
        self.closing=True
        # Exclusive marker authority is a preflight for every borrowed-state
        # mutation, including hooks/checkpoint cleanup on the first close.
        if not self.pending_marker or vars(self.model).get(MARKER)is not self:
            raise BaseExceptionGroup('ZImage attachment cleanup pending',
                [RuntimeError('foreign ZImage attachment marker retained')])
        errors=[]
        try:self._assert_handle_container()
        except BaseException as error:errors.append(error)
        else:self.pending_hooks=self._run(self.pending_hooks,self._remove_hook,errors)
        if not self.pending_hooks and self.pending_container:
            try:self._assert_handle_container()
            except BaseException as error:errors.append(error)
            else:self.handle_container.clear();self.handle_items=();self.pending_container=False
        def checkpoint(value):
            module,key,had,old,has_owned,owned=value
            if (key in vars(module))!=has_owned or vars(module).get(key)is not owned:
                raise RuntimeError('foreign native checkpoint attribute retained')
            if had:setattr(module,key,old)
            else:vars(module).pop(key,None)
        self.pending_checkpoint=self._run(self.pending_checkpoint,checkpoint,errors)
        if not self.pending_hooks and not self.pending_checkpoint and not self.pending_container:
            if not self.pending_marker or vars(self.model).get(MARKER)is not self:
                errors.append(RuntimeError('foreign ZImage attachment marker retained'))
            else:
                def flag(value):
                    parameter,original=value
                    if parameter.requires_grad is not self.owned_flags[id(parameter)]:
                        raise RuntimeError('foreign native gradient flag retained')
                    parameter.requires_grad_(original)
                def mode(value):
                    module,original=value
                    if module.training is not self.owned_modes[id(module)]:
                        raise RuntimeError('foreign native training mode retained')
                    module.training=original
                self.pending_flags=self._run(self.pending_flags,flag,errors)
                self.pending_modes=self._run(self.pending_modes,mode,errors)
                # Keep ownership through every retryable operation. A failure
                # must not admit another owner before this owner finishes.
                if not self.pending_flags and not self.pending_modes:
                    if vars(self.model).get(MARKER)is not self:
                        errors.append(RuntimeError('foreign ZImage attachment marker retained'))
                    else:delattr(self.model,MARKER);self.pending_marker=False
        if errors:raise BaseExceptionGroup('ZImage attachment cleanup pending',errors)
        if any((self.pending_hooks,self.pending_checkpoint,self.pending_flags,self.pending_modes))or self.pending_marker or self.pending_container:
            raise RuntimeError('ZImage cleanup still pending')
        self.closed,self.closing=True,False

class Owner:
    def __init__(self):self.closed=self.closing=False;self.attachment=None
    def require_open(self):
        if self.closed or self.closing:raise RuntimeError('ZImage owner closed/cleanup pending')
        if self.attachment is not None:self.attachment.require_active()
    def close(self):
        if self.closed:return
        self.closing=True
        if self.attachment is not None:self.attachment.close()
        self.closed,self.closing=True,False
    def __enter__(self):self.require_open();return self
    def __exit__(self,kind,primary,traceback):
        try:self.close()
        except BaseException as cleanup:
            if primary is not None:raise BaseExceptionGroup('ZImage work and cleanup failed',[primary,cleanup])
            raise
