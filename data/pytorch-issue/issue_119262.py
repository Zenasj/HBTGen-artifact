#0  __futex_abstimed_wait_common64 (private=792123632, cancel=true, abstime=0x1538ffda5140, op=137, expected=0, futex_word=0x5555558e4a0c <_PyRuntime+428>) at ./nptl/futex-internal.c:57
#1  __futex_abstimed_wait_common (cancel=true, private=792123632, abstime=0x1538ffda5140, clockid=-1023316592, expected=0, futex_word=0x5555558e4a0c <_PyRuntime+428>) at ./nptl/futex-internal.c:87
#2  __GI___futex_abstimed_wait_cancelable64 (futex_word=futex_word@entry=0x5555558e4a0c <_PyRuntime+428>, expected=expected@entry=0, clockid=clockid@entry=1, abstime=abstime@entry=0x1538ffda5140, private=private@entry=0) at ./nptl/futex-internal.c:139
#3  0x0000155555273f1b in __pthread_cond_wait_common (abstime=0x1538ffda5140, clockid=1, mutex=0x5555558e4a10 <_PyRuntime+432>, cond=0x5555558e49e0 <_PyRuntime+384>) at ./nptl/pthread_cond_wait.c:503
#4  ___pthread_cond_timedwait64 (cond=cond@entry=0x5555558e49e0 <_PyRuntime+384>, mutex=mutex@entry=0x5555558e4a10 <_PyRuntime+432>, abstime=abstime@entry=0x1538ffda5140) at ./nptl/pthread_cond_wait.c:652
#5  0x000055555566f8cd in PyCOND_TIMEDWAIT (us=<optimized out>, mut=<optimized out>, cond=0x5555558e49e0 <_PyRuntime+384>) at python-3.10.12/Python/condvar.h:73
#6  take_gil (tstate=0x153648000b90) at python-3.10.12/Python/ceval_gil.h:256
#7  0x00005555556b3332 in PyEval_RestoreThread (tstate=0x153648000b90) at python-3.10.12/Python/ceval.c:547
#8  0x000015550fa4057b in THPVariable_clear(THPVariable*) () from /torch/lib/libtorch_python.so
#9  0x000015550fa40b45 in THPVariable_subclass_dealloc(_object*) () from /torch/lib/libtorch_python.so
#10 0x000015550f98fe36 in (anonymous namespace)::ConcretePyInterpreterVTable::decref(_object*, bool) const () from /torch/lib/libtorch_python.so
#11 0x000015551279bbe3 in c10::impl::PyObjectSlot::destroy_pyobj_if_needed() () from /torch/lib/libc10.so
#12 0x0000155512789dbd in c10::TensorImpl::~TensorImpl() () from /torch/lib/libc10.so
#13 0x000015551278a12d in c10::TensorImpl::~TensorImpl() () from /torch/lib/libc10.so
#14 0x00001555080db703 in at::autocast::clear_cache() () from /torch/lib/libtorch_cpu.so
#15 0x000015550f9c548c in torch::autograd::clear_autocast_cache(_object*, _object*) () from /torch/lib/libtorch_python.so
#16 0x0000555555696362 in cfunction_vectorcall_NOARGS (func=0x155554505670, args=<optimized out>, nargsf=<optimized out>, kwnames=<optimized out>) at python-3.10.12/Objects/methodobject.c:489
#17 0x000055555568d142 in _PyObject_VectorcallTstate (kwnames=0x0, nargsf=<optimized out>, args=0x153ae2fc01d0, callable=0x155554505670, tstate=0x153648000b90) at python-3.10.12/Include/cpython/abstract.h:114
#18 PyObject_Vectorcall (kwnames=0x0, nargsf=<optimized out>, args=0x153ae2fc01d0, callable=0x155554505670) at python-3.10.12/Include/cpython/abstract.h:123
#19 call_function (kwnames=0x0, oparg=<optimized out>, pp_stack=<synthetic pointer>, trace_info=0x1538ffda5590, tstate=<optimized out>) at python-3.10.12/Python/ceval.c:5893
#20 _PyEval_EvalFrameDefault (tstate=<optimized out>, f=0x153ae2fc0040, throwflag=<optimized out>) at python-3.10.12/Python/ceval.c:4181
#21 0x00005555556a44f2 in _PyEval_EvalFrame (throwflag=0, f=0x153ae2fc0040, tstate=0x153648000b90) at python-3.10.12/Include/internal/pycore_ceval.h:46
#22 _PyEval_Vector (kwnames=<optimized out>, argcount=<optimized out>, args=0x55559567cb90, locals=0x0, con=0x1554c33a7a40, tstate=0x153648000b90) at python-3.10.12/Python/ceval.c:5067
#23 _PyFunction_Vectorcall (kwnames=<optimized out>, nargsf=<optimized out>, stack=0x55559567cb90, func=0x1554c33a7a30) at python-3.10.12/Objects/call.c:342
#24 _PyObject_VectorcallTstate (kwnames=<optimized out>, nargsf=<optimized out>, args=0x55559567cb90, callable=0x1554c33a7a30, tstate=0x153648000b90) at python-3.10.12/Include/cpython/abstract.h:114
#25 method_vectorcall (method=<optimized out>, args=0x55559567cb98, nargsf=<optimized out>, kwnames=<optimized out>) at python-3.10.12/Objects/classobject.c:53
#26 0x000055555568d142 in _PyObject_VectorcallTstate (kwnames=0x0, nargsf=<optimized out>, args=0x55559567cb98, callable=0x153b38ad6680, tstate=0x153648000b90) at python-3.10.12/Include/cpython/abstract.h:114
#27 PyObject_Vectorcall (kwnames=0x0, nargsf=<optimized out>, args=0x55559567cb98, callable=0x153b38ad6680) at python-3.10.12/Include/cpython/abstract.h:123
#28 call_function (kwnames=0x0, oparg=<optimized out>, pp_stack=<synthetic pointer>, trace_info=0x1538ffda57a0, tstate=<optimized out>) at python-3.10.12/Python/ceval.c:5893
#29 _PyEval_EvalFrameDefault (tstate=<optimized out>, f=0x55559567ca00, throwflag=<optimized out>) at python-3.10.12/Python/ceval.c:4181
#30 0x00005555556a44f2 in _PyEval_EvalFrame (throwflag=0, f=0x55559567ca00, tstate=0x153648000b90) at python-3.10.12/Include/internal/pycore_ceval.h:46
#31 _PyEval_Vector (kwnames=<optimized out>, argcount=<optimized out>, args=0x1538ffda5a10, locals=0x0, con=0x1554c3114680, tstate=0x153648000b90) at python-3.10.12/Python/ceval.c:5067
#32 _PyFunction_Vectorcall (kwnames=<optimized out>, nargsf=<optimized out>, stack=0x1538ffda5a10, func=0x1554c3114670) at python-3.10.12/Objects/call.c:342
#33 _PyObject_VectorcallTstate (kwnames=<optimized out>, nargsf=<optimized out>, args=0x1538ffda5a10, callable=0x1554c3114670, tstate=0x153648000b90) at python-3.10.12/Include/cpython/abstract.h:114
#34 method_vectorcall (method=<optimized out>, args=0x1538ffda5a18, nargsf=<optimized out>, kwnames=<optimized out>) at python-3.10.12/Objects/classobject.c:53
#35 0x000055555572a572 in _PyObject_VectorcallTstate (tstate=0x153648000b90, callable=0x153b38ad6280, args=<optimized out>, nargsf=<optimized out>, kwnames=<optimized out>) at python-3.10.12/Include/cpython/abstract.h:114
#36 0x000055555568bd74 in PyObject_Vectorcall (kwnames=0x0, nargsf=9223372036854775811, args=0x1538ffda5a18, callable=0x153b38ad6280) at python-3.10.12/Include/cpython/abstract.h:123
#37 _PyEval_EvalFrameDefault (tstate=<optimized out>, f=0x15372eae5360, throwflag=<optimized out>) at python-3.10.12/Python/ceval.c:4113
#38 0x000055555569899c in _PyEval_EvalFrame (throwflag=0, f=0x15372eae5360, tstate=0x153648000b90) at python-3.10.12/Include/internal/pycore_ceval.h:46
#39 _PyEval_Vector (kwnames=<optimized out>, argcount=<optimized out>, args=<optimized out>, locals=0x0, con=0x153ae302a570, tstate=0x153648000b90) at python-3.10.12/Python/ceval.c:5067
#40 _PyFunction_Vectorcall (func=0x153ae302a560, stack=<optimized out>, nargsf=<optimized out>, kwnames=<optimized out>) at python-3.10.12/Objects/call.c:342
#41 0x000055555568b2b0 in do_call_core (kwdict=0x0, callargs=0x153ae32035c0, func=0x153ae302a560, trace_info=0x1538ffda5b80, tstate=<optimized out>) at python-3.10.12/Python/ceval.c:5945
#42 _PyEval_EvalFrameDefault (tstate=<optimized out>, f=0x5555d155be30, throwflag=<optimized out>) at python-3.10.12/Python/ceval.c:4277
#43 0x000055555569899c in _PyEval_EvalFrame (throwflag=0, f=0x5555d155be30, tstate=0x153648000b90) at python-3.10.12/Include/internal/pycore_ceval.h:46
#44 _PyEval_Vector (kwnames=<optimized out>, argcount=<optimized out>, args=<optimized out>, locals=0x0, con=0x153ae3028050, tstate=0x153648000b90) at python-3.10.12/Python/ceval.c:5067
#45 _PyFunction_Vectorcall (func=0x153ae3028040, stack=<optimized out>, nargsf=<optimized out>, kwnames=<optimized out>) at python-3.10.12/Objects/call.c:342
#46 0x0000555555698178 in _PyObject_VectorcallTstate (kwnames=0x0, nargsf=1, args=0x1538ffda5cb0, callable=0x153ae3028040, tstate=0x153648000b90) at python-3.10.12/Include/cpython/abstract.h:114
#47 object_vacall (tstate=0x153648000b90, base=<optimized out>, callable=0x153ae3028040, vargs=0x1538ffda5d20) at python-3.10.12/Objects/call.c:734
#48 0x000055555572dd06 in PyObject_CallFunctionObjArgs (callable=<optimized out>) at python-3.10.12/Objects/call.c:841
#49 0x000015550fa162e8 in torch::autograd::PySavedVariableHooks::call_unpack_hook() () from /torch/lib/libtorch_python.so
#50 0x000015550b81e924 in torch::autograd::SavedVariable::unpack(std::shared_ptr<torch::autograd::Node>) const () from /torch/lib/libtorch_cpu.so
#51 0x000015550aa134df in torch::autograd::generated::HardtanhBackward0::apply(std::vector<at::Tensor, std::allocator<at::Tensor> >&&) () from /torch/lib/libtorch_cpu.so
#52 0x000015550b7e54eb in torch::autograd::Node::operator()(std::vector<at::Tensor, std::allocator<at::Tensor> >&&) () from /torch/lib/libtorch_cpu.so
#53 0x000015550b7debea in torch::autograd::Engine::evaluate_function(std::shared_ptr<torch::autograd::GraphTask>&, torch::autograd::Node*, torch::autograd::InputBuffer&, std::shared_ptr<torch::autograd::ReadyQueue> const&) () from /torch/lib/libtorch_cpu.so
#54 0x000015550b7dfed0 in torch::autograd::Engine::thread_main(std::shared_ptr<torch::autograd::GraphTask> const&) () from /torch/lib/libtorch_cpu.so
#55 0x000015550b7d68da in torch::autograd::Engine::thread_init(int, std::shared_ptr<torch::autograd::ReadyQueue> const&, bool) () from /torch/lib/libtorch_cpu.so
#56 0x000015550fa1b600 in torch::autograd::python::PythonEngine::thread_init(int, std::shared_ptr<torch::autograd::ReadyQueue> const&, bool) () from /torch/lib/libtorch_python.so
#57 0x00001555122f0e95 in std::execute_native_thread_routine (__p=<optimized out>) at ../../../../../libstdc++-v3/src/c++11/thread.cc:104
#58 0x0000155555274b43 in start_thread (arg=<optimized out>) at ./nptl/pthread_create.c:442
#59 0x0000155555306a00 in clone3 () at ../sysdeps/unix/sysv/linux/x86_64/clone3.S:81

#0  futex_wait (private=0, expected=2, futex_word=0x15550f1ccc00 <at::autocast::(anonymous namespace)::cached_casts_mutex>) at ../sysdeps/nptl/futex-internal.h:146
#1  __GI___lll_lock_wait (futex=futex@entry=0x15550f1ccc00 <at::autocast::(anonymous namespace)::cached_casts_mutex>, private=0) at ./nptl/lowlevellock.c:49
#2  0x0000155555278082 in lll_mutex_lock_optimized (mutex=0x15550f1ccc00 <at::autocast::(anonymous namespace)::cached_casts_mutex>) at ./nptl/pthread_mutex_lock.c:48
#3  ___pthread_mutex_lock (mutex=0x15550f1ccc00 <at::autocast::(anonymous namespace)::cached_casts_mutex>) at ./nptl/pthread_mutex_lock.c:93
#4  0x00001555080db616 in at::autocast::clear_cache() () from /torch/lib/libtorch_cpu.so
#5  0x000015550f9c548c in torch::autograd::clear_autocast_cache(_object*, _object*) () from /torch/lib/libtorch_python.so
#6  0x0000555555696362 in cfunction_vectorcall_NOARGS (func=0x155554505670, args=<optimized out>, nargsf=<optimized out>, kwnames=<optimized out>) at python-3.10.12/Objects/methodobject.c:489
#7  0x000055555568d142 in _PyObject_VectorcallTstate (kwnames=0x0, nargsf=<optimized out>, args=0x555572b04020, callable=0x155554505670, tstate=0x55555590a0c0) at python-3.10.12/Include/cpython/abstract.h:114
#8  PyObject_Vectorcall (kwnames=0x0, nargsf=<optimized out>, args=0x555572b04020, callable=0x155554505670) at python-3.10.12/Include/cpython/abstract.h:123
#9  call_function (kwnames=0x0, oparg=<optimized out>, pp_stack=<synthetic pointer>, trace_info=0x7fffffff61c0, tstate=<optimized out>) at python-3.10.12/Python/ceval.c:5893
#10 _PyEval_EvalFrameDefault (tstate=<optimized out>, f=0x555572b03e90, throwflag=<optimized out>) at python-3.10.12/Python/ceval.c:4181
#11 0x00005555556a44f2 in _PyEval_EvalFrame (throwflag=0, f=0x555572b03e90, tstate=0x55555590a0c0) at python-3.10.12/Include/internal/pycore_ceval.h:46
#12 _PyEval_Vector (kwnames=<optimized out>, argcount=<optimized out>, args=0x5555d13839b0, locals=0x0, con=0x1554c33a7a40, tstate=0x55555590a0c0) at python-3.10.12/Python/ceval.c:5067
#13 _PyFunction_Vectorcall (kwnames=<optimized out>, nargsf=<optimized out>, stack=0x5555d13839b0, func=0x1554c33a7a30) at python-3.10.12/Objects/call.c:342
#14 _PyObject_VectorcallTstate (kwnames=<optimized out>, nargsf=<optimized out>, args=0x5555d13839b0, callable=0x1554c33a7a30, tstate=0x55555590a0c0) at python-3.10.12/Include/cpython/abstract.h:114
#15 method_vectorcall (method=<optimized out>, args=0x5555d13839b8, nargsf=<optimized out>, kwnames=<optimized out>) at python-3.10.12/Objects/classobject.c:53
#16 0x000055555568d142 in _PyObject_VectorcallTstate (kwnames=0x0, nargsf=<optimized out>, args=0x5555d13839b8, callable=0x153b38ad58c0, tstate=0x55555590a0c0) at python-3.10.12/Include/cpython/abstract.h:114
#17 PyObject_Vectorcall (kwnames=0x0, nargsf=<optimized out>, args=0x5555d13839b8, callable=0x153b38ad58c0) at python-3.10.12/Include/cpython/abstract.h:123
#18 call_function (kwnames=0x0, oparg=<optimized out>, pp_stack=<synthetic pointer>, trace_info=0x7fffffff63d0, tstate=<optimized out>) at python-3.10.12/Python/ceval.c:5893
#19 _PyEval_EvalFrameDefault (tstate=<optimized out>, f=0x5555d1383820, throwflag=<optimized out>) at python-3.10.12/Python/ceval.c:4181
#20 0x00005555556a44f2 in _PyEval_EvalFrame (throwflag=0, f=0x5555d1383820, tstate=0x55555590a0c0) at python-3.10.12/Include/internal/pycore_ceval.h:46
#21 _PyEval_Vector (kwnames=<optimized out>, argcount=<optimized out>, args=0x7fffffff6640, locals=0x0, con=0x1554c3114680, tstate=0x55555590a0c0) at python-3.10.12/Python/ceval.c:5067
#22 _PyFunction_Vectorcall (kwnames=<optimized out>, nargsf=<optimized out>, stack=0x7fffffff6640, func=0x1554c3114670) at python-3.10.12/Objects/call.c:342
#23 _PyObject_VectorcallTstate (kwnames=<optimized out>, nargsf=<optimized out>, args=0x7fffffff6640, callable=0x1554c3114670, tstate=0x55555590a0c0) at python-3.10.12/Include/cpython/abstract.h:114
#24 method_vectorcall (method=<optimized out>, args=0x7fffffff6648, nargsf=<optimized out>, kwnames=<optimized out>) at python-3.10.12/Objects/classobject.c:53
#25 0x000055555572a572 in _PyObject_VectorcallTstate (tstate=0x55555590a0c0, callable=0x153b02f55f80, args=<optimized out>, nargsf=<optimized out>, kwnames=<optimized out>) at python-3.10.12/Include/cpython/abstract.h:114
#26 0x000055555568bd74 in PyObject_Vectorcall (kwnames=0x0, nargsf=9223372036854775811, args=0x7fffffff6648, callable=0x153b02f55f80) at python-3.10.12/Include/cpython/abstract.h:123
#27 _PyEval_EvalFrameDefault (tstate=<optimized out>, f=0x5555a4559930, throwflag=<optimized out>) at python-3.10.12/Python/ceval.c:4113
#28 0x000055555569899c in _PyEval_EvalFrame (throwflag=0, f=0x5555a4559930, tstate=0x55555590a0c0) at python-3.10.12/Include/internal/pycore_ceval.h:46
#29 _PyEval_Vector (kwnames=<optimized out>, argcount=<optimized out>, args=<optimized out>, locals=0x0, con=0x153ae3028290, tstate=0x55555590a0c0) at python-3.10.12/Python/ceval.c:5067
#30 _PyFunction_Vectorcall (func=0x153ae3028280, stack=<optimized out>, nargsf=<optimized out>, kwnames=<optimized out>) at python-3.10.12/Objects/call.c:342
#31 0x000055555568b2b0 in do_call_core (kwdict=0x0, callargs=0x153b036a63c0, func=0x153ae3028280, trace_info=0x7fffffff67b0, tstate=<optimized out>) at python-3.10.12/Python/ceval.c:5945
#32 _PyEval_EvalFrameDefault (tstate=<optimized out>, f=0x5555a1cc8fe0, throwflag=<optimized out>) at python-3.10.12/Python/ceval.c:4277
#33 0x000055555569899c in _PyEval_EvalFrame (throwflag=0, f=0x5555a1cc8fe0, tstate=0x55555590a0c0) at python-3.10.12/Include/internal/pycore_ceval.h:46
#34 _PyEval_Vector (kwnames=<optimized out>, argcount=<optimized out>, args=<optimized out>, locals=0x0, con=0x153ae3029e20, tstate=0x55555590a0c0) at python-3.10.12/Python/ceval.c:5067
#35 _PyFunction_Vectorcall (func=0x153ae3029e10, stack=<optimized out>, nargsf=<optimized out>, kwnames=<optimized out>) at python-3.10.12/Objects/call.c:342
#36 0x0000555555698178 in _PyObject_VectorcallTstate (kwnames=0x0, nargsf=1, args=0x7fffffff68e0, callable=0x153ae3029e10, tstate=0x55555590a0c0) at python-3.10.12/Include/cpython/abstract.h:114
#37 object_vacall (tstate=0x55555590a0c0, base=<optimized out>, callable=0x153ae3029e10, vargs=0x7fffffff6950) at python-3.10.12/Objects/call.c:734
#38 0x000055555572dd06 in PyObject_CallFunctionObjArgs (callable=<optimized out>) at python-3.10.12/Objects/call.c:841
#39 0x000015550fa162e8 in torch::autograd::PySavedVariableHooks::call_unpack_hook() () from /torch/lib/libtorch_python.so
#40 0x000015550b81e924 in torch::autograd::SavedVariable::unpack(std::shared_ptr<torch::autograd::Node>) const () from /torch/lib/libtorch_cpu.so
#41 0x000015550aa134df in torch::autograd::generated::HardtanhBackward0::apply(std::vector<at::Tensor, std::allocator<at::Tensor> >&&) () from /torch/lib/libtorch_cpu.so
#42 0x000015550b7e54eb in torch::autograd::Node::operator()(std::vector<at::Tensor, std::allocator<at::Tensor> >&&) () from /torch/lib/libtorch_cpu.so
#43 0x000015550b7debea in torch::autograd::Engine::evaluate_function(std::shared_ptr<torch::autograd::GraphTask>&, torch::autograd::Node*, torch::autograd::InputBuffer&, std::shared_ptr<torch::autograd::ReadyQueue> const&) () from /torch/lib/libtorch_cpu.so
#44 0x000015550b7dfed0 in torch::autograd::Engine::thread_main(std::shared_ptr<torch::autograd::GraphTask> const&) () from /torch/lib/libtorch_cpu.so
#45 0x000015550b7da965 in torch::autograd::Engine::execute_with_graph_task(std::shared_ptr<torch::autograd::GraphTask> const&, std::shared_ptr<torch::autograd::Node>, torch::autograd::InputBuffer&&) () from /torch/lib/libtorch_cpu.so
#46 0x000015550fa1bb39 in torch::autograd::python::PythonEngine::execute_with_graph_task(std::shared_ptr<torch::autograd::GraphTask> const&, std::shared_ptr<torch::autograd::Node>, torch::autograd::InputBuffer&&) () from /torch/lib/libtorch_python.so
#47 0x000015550b7dd755 in torch::autograd::Engine::execute(std::vector<torch::autograd::Edge, std::allocator<torch::autograd::Edge> > const&, std::vector<at::Tensor, std::allocator<at::Tensor> > const&, bool, bool, bool, std::vector<torch::autograd::Edge, std::allocator<torch::autograd::Edge> > const&) () from /torch/lib/libtorch_cpu.so
#48 0x000015550fa1ba9e in torch::autograd::python::PythonEngine::execute(std::vector<torch::autograd::Edge, std::allocator<torch::autograd::Edge> > const&, std::vector<at::Tensor, std::allocator<at::Tensor> > const&, bool, bool, bool, std::vector<torch::autograd::Edge, std::allocator<torch::autograd::Edge> > const&) () from /torch/lib/libtorch_python.so
#49 0x000015550fa1a2b0 in THPEngine_run_backward(_object*, _object*, _object*) () from /torch/lib/libtorch_python.so
#50 0x0000555555698516 in cfunction_call (func=0x153b03466ed0, args=<optimized out>, kwargs=<optimized out>) at python-3.10.12/Objects/methodobject.c:543
#51 0x0000555555691a6b in _PyObject_MakeTpCall (tstate=0x55555590a0c0, callable=0x153b03466ed0, args=<optimized out>, nargs=5, keywords=0x1554c30f1dc0) at python-3.10.12/Objects/call.c:215
#52 0x000055555568dc39 in _PyObject_VectorcallTstate (kwnames=0x1554c30f1dc0, nargsf=<optimized out>, args=<optimized out>, callable=0x153b03466ed0, tstate=<optimized out>) at python-3.10.12/Include/cpython/abstract.h:112
#53 _PyObject_VectorcallTstate (kwnames=0x1554c30f1dc0, nargsf=<optimized out>, args=<optimized out>, callable=0x153b03466ed0, tstate=<optimized out>) at python-3.10.12/Include/cpython/abstract.h:99
#54 PyObject_Vectorcall (kwnames=0x1554c30f1dc0, nargsf=<optimized out>, args=<optimized out>, callable=0x153b03466ed0) at python-3.10.12/Include/cpython/abstract.h:123
#55 call_function (kwnames=0x1554c30f1dc0, oparg=<optimized out>, pp_stack=<synthetic pointer>, trace_info=0x7fffffff8240, tstate=<optimized out>) at python-3.10.12/Python/ceval.c:5893
#56 _PyEval_EvalFrameDefault (tstate=<optimized out>, f=0x153ae7d47c40, throwflag=<optimized out>) at python-3.10.12/Python/ceval.c:4231
#57 0x000055555569899c in _PyEval_EvalFrame (throwflag=0, f=0x153ae7d47c40, tstate=0x55555590a0c0) at python-3.10.12/Include/internal/pycore_ceval.h:46
#58 _PyEval_Vector (kwnames=<optimized out>, argcount=<optimized out>, args=<optimized out>, locals=0x0, con=0x1554c2e7b530, tstate=0x55555590a0c0) at python-3.10.12/Python/ceval.c:5067
#59 _PyFunction_Vectorcall (func=0x1554c2e7b520, stack=<optimized out>, nargsf=<optimized out>, kwnames=<optimized out>) at python-3.10.12/Objects/call.c:342
#60 0x00005555556898fa in _PyObject_VectorcallTstate (kwnames=0x1554c346f0d0, nargsf=<optimized out>, args=<optimized out>, callable=0x1554c2e7b520, tstate=0x55555590a0c0) at python-3.10.12/Include/cpython/abstract.h:114
#61 PyObject_Vectorcall (kwnames=0x1554c346f0d0, nargsf=<optimized out>, args=<optimized out>, callable=0x1554c2e7b520) at python-3.10.12/Include/cpython/abstract.h:123
#62 call_function (kwnames=0x1554c346f0d0, oparg=<optimized out>, pp_stack=<synthetic pointer>, trace_info=0x7fffffff83f0, tstate=<optimized out>) at python-3.10.12/Python/ceval.c:5893
#63 _PyEval_EvalFrameDefault (tstate=<optimized out>, f=0x153ae2f63490, throwflag=<optimized out>) at python-3.10.12/Python/ceval.c:4231
#64 0x000055555569899c in _PyEval_EvalFrame (throwflag=0, f=0x153ae2f63490, tstate=0x55555590a0c0) at python-3.10.12/Include/internal/pycore_ceval.h:46
#65 _PyEval_Vector (kwnames=<optimized out>, argcount=<optimized out>, args=<optimized out>, locals=0x0, con=0x1554c34d52e0, tstate=0x55555590a0c0) at python-3.10.12/Python/ceval.c:5067
#66 _PyFunction_Vectorcall (func=0x1554c34d52d0, stack=<optimized out>, nargsf=<optimized out>, kwnames=<optimized out>) at python-3.10.12/Objects/call.c:342
#67 0x0000555555688c5c in _PyObject_VectorcallTstate (kwnames=0x0, nargsf=<optimized out>, args=0x5555ae6dc208, callable=0x1554c34d52d0, tstate=0x55555590a0c0) at python-3.10.12/Include/cpython/abstract.h:114
#68 PyObject_Vectorcall (kwnames=0x0, nargsf=<optimized out>, args=0x5555ae6dc208, callable=0x1554c34d52d0) at python-3.10.12/Include/cpython/abstract.h:123
#69 call_function (kwnames=0x0, oparg=<optimized out>, pp_stack=<synthetic pointer>, trace_info=0x7fffffff85a0, tstate=<optimized out>) at python-3.10.12/Python/ceval.c:5893
#70 _PyEval_EvalFrameDefault (tstate=<optimized out>, f=0x5555ae6dbfa0, throwflag=<optimized out>) at python-3.10.12/Python/ceval.c:4198
#71 0x000055555569899c in _PyEval_EvalFrame (throwflag=0, f=0x5555ae6dbfa0, tstate=0x55555590a0c0) at python-3.10.12/Include/internal/pycore_ceval.h:46
#72 _PyEval_Vector (kwnames=<optimized out>, argcount=<optimized out>, args=<optimized out>, locals=0x0, con=0x1551df3a9130, tstate=0x55555590a0c0) at python-3.10.12/Python/ceval.c:5067
#73 _PyFunction_Vectorcall (func=0x1551df3a9120, stack=<optimized out>, nargsf=<optimized out>, kwnames=<optimized out>) at python-3.10.12/Objects/call.c:342
#74 0x0000555555688850 in _PyObject_VectorcallTstate (kwnames=0x0, nargsf=<optimized out>, args=0x5555a551c548, callable=0x1551df3a9120, tstate=0x55555590a0c0) at python-3.10.12/Include/cpython/abstract.h:114
#75 PyObject_Vectorcall (kwnames=0x0, nargsf=<optimized out>, args=0x5555a551c548, callable=0x1551df3a9120) at python-3.10.12/Include/cpython/abstract.h:123
#76 call_function (kwnames=0x0, oparg=<optimized out>, pp_stack=<synthetic pointer>, trace_info=0x7fffffff8750, tstate=<optimized out>) at python-3.10.12/Python/ceval.c:5893
#77 _PyEval_EvalFrameDefault (tstate=<optimized out>, f=0x5555a551c320, throwflag=<optimized out>) at python-3.10.12/Python/ceval.c:4213
#78 0x000055555569899c in _PyEval_EvalFrame (throwflag=0, f=0x5555a551c320, tstate=0x55555590a0c0) at python-3.10.12/Include/internal/pycore_ceval.h:46
#79 _PyEval_Vector (kwnames=<optimized out>, argcount=<optimized out>, args=<optimized out>, locals=0x0, con=0x1551df3a8f80, tstate=0x55555590a0c0) at python-3.10.12/Python/ceval.c:5067
#80 _PyFunction_Vectorcall (func=0x1551df3a8f70, stack=<optimized out>, nargsf=<optimized out>, kwnames=<optimized out>) at python-3.10.12/Objects/call.c:342
#81 0x00005555556898fa in _PyObject_VectorcallTstate (kwnames=0x155554a17700, nargsf=<optimized out>, args=<optimized out>, callable=0x1551df3a8f70, tstate=0x55555590a0c0) at python-3.10.12/Include/cpython/abstract.h:114
#82 PyObject_Vectorcall (kwnames=0x155554a17700, nargsf=<optimized out>, args=<optimized out>, callable=0x1551df3a8f70) at python-3.10.12/Include/cpython/abstract.h:123
#83 call_function (kwnames=0x155554a17700, oparg=<optimized out>, pp_stack=<synthetic pointer>, trace_info=0x7fffffff8900, tstate=<optimized out>) at python-3.10.12/Python/ceval.c:5893
#84 _PyEval_EvalFrameDefault (tstate=<optimized out>, f=0x55558ca972f0, throwflag=<optimized out>) at python-3.10.12/Python/ceval.c:4231
#85 0x000055555569899c in _PyEval_EvalFrame (throwflag=0, f=0x55558ca972f0, tstate=0x55555590a0c0) at python-3.10.12/Include/internal/pycore_ceval.h:46
#86 _PyEval_Vector (kwnames=<optimized out>, argcount=<optimized out>, args=<optimized out>, locals=0x0, con=0x1551de1d3650, tstate=0x55555590a0c0) at python-3.10.12/Python/ceval.c:5067
#87 _PyFunction_Vectorcall (func=0x1551de1d3640, stack=<optimized out>, nargsf=<optimized out>, kwnames=<optimized out>) at python-3.10.12/Objects/call.c:342
#88 0x0000555555688850 in _PyObject_VectorcallTstate (kwnames=0x0, nargsf=<optimized out>, args=0x155168ffb9f0, callable=0x1551de1d3640, tstate=0x55555590a0c0) at python-3.10.12/Include/cpython/abstract.h:114
#89 PyObject_Vectorcall (kwnames=0x0, nargsf=<optimized out>, args=0x155168ffb9f0, callable=0x1551de1d3640) at python-3.10.12/Include/cpython/abstract.h:123
#90 call_function (kwnames=0x0, oparg=<optimized out>, pp_stack=<synthetic pointer>, trace_info=0x7fffffff8ab0, tstate=<optimized out>) at python-3.10.12/Python/ceval.c:5893
#91 _PyEval_EvalFrameDefault (tstate=<optimized out>, f=0x155168ffb840, throwflag=<optimized out>) at python-3.10.12/Python/ceval.c:4213
#92 0x000055555569899c in _PyEval_EvalFrame (throwflag=0, f=0x155168ffb840, tstate=0x55555590a0c0) at python-3.10.12/Include/internal/pycore_ceval.h:46
#93 _PyEval_Vector (kwnames=<optimized out>, argcount=<optimized out>, args=<optimized out>, locals=0x0, con=0x1551de1d3800, tstate=0x55555590a0c0) at python-3.10.12/Python/ceval.c:5067
#94 _PyFunction_Vectorcall (func=0x1551de1d37f0, stack=<optimized out>, nargsf=<optimized out>, kwnames=<optimized out>) at python-3.10.12/Objects/call.c:342
#95 0x0000555555688850 in _PyObject_VectorcallTstate (kwnames=0x0, nargsf=<optimized out>, args=0x153c96c27df0, callable=0x1551de1d37f0, tstate=0x55555590a0c0) at python-3.10.12/Include/cpython/abstract.h:114
#96 PyObject_Vectorcall (kwnames=0x0, nargsf=<optimized out>, args=0x153c96c27df0, callable=0x1551de1d37f0) at python-3.10.12/Include/cpython/abstract.h:123
#97 call_function (kwnames=0x0, oparg=<optimized out>, pp_stack=<synthetic pointer>, trace_info=0x7fffffff8c60, tstate=<optimized out>) at python-3.10.12/Python/ceval.c:5893
#98 _PyEval_EvalFrameDefault (tstate=<optimized out>, f=0x153c96c27c50, throwflag=<optimized out>) at python-3.10.12/Python/ceval.c:4213
#99 0x000055555569899c in _PyEval_EvalFrame (throwflag=0, f=0x153c96c27c50, tstate=0x55555590a0c0) at python-3.10.12/Include/internal/pycore_ceval.h:46
#100 _PyEval_Vector (kwnames=<optimized out>, argcount=<optimized out>, args=<optimized out>, locals=0x0, con=0x1551de1d39b0, tstate=0x55555590a0c0) at python-3.10.12/Python/ceval.c:5067
#101 _PyFunction_Vectorcall (func=0x1551de1d39a0, stack=<optimized out>, nargsf=<optimized out>, kwnames=<optimized out>) at python-3.10.12/Objects/call.c:342
#102 0x0000555555688850 in _PyObject_VectorcallTstate (kwnames=0x0, nargsf=<optimized out>, args=0x555563a943e0, callable=0x1551de1d39a0, tstate=0x55555590a0c0) at python-3.10.12/Include/cpython/abstract.h:114
#103 PyObject_Vectorcall (kwnames=0x0, nargsf=<optimized out>, args=0x555563a943e0, callable=0x1551de1d39a0) at python-3.10.12/Include/cpython/abstract.h:123
#104 call_function (kwnames=0x0, oparg=<optimized out>, pp_stack=<synthetic pointer>, trace_info=0x7fffffff8e10, tstate=<optimized out>) at python-3.10.12/Python/ceval.c:5893
#105 _PyEval_EvalFrameDefault (tstate=<optimized out>, f=0x555563a94210, throwflag=<optimized out>) at python-3.10.12/Python/ceval.c:4213
#106 0x000055555569899c in _PyEval_EvalFrame (throwflag=0, f=0x555563a94210, tstate=0x55555590a0c0) at python-3.10.12/Include/internal/pycore_ceval.h:46
#107 _PyEval_Vector (kwnames=<optimized out>, argcount=<optimized out>, args=<optimized out>, locals=0x0, con=0x1551de1d3ad0, tstate=0x55555590a0c0) at python-3.10.12/Python/ceval.c:5067
#108 _PyFunction_Vectorcall (func=0x1551de1d3ac0, stack=<optimized out>, nargsf=<optimized out>, kwnames=<optimized out>) at python-3.10.12/Objects/call.c:342
#109 0x00005555556898fa in _PyObject_VectorcallTstate (kwnames=0x155554a2c7c0, nargsf=<optimized out>, args=<optimized out>, callable=0x1551de1d3ac0, tstate=0x55555590a0c0) at python-3.10.12/Include/cpython/abstract.h:114
#110 PyObject_Vectorcall (kwnames=0x155554a2c7c0, nargsf=<optimized out>, args=<optimized out>, callable=0x1551de1d3ac0) at python-3.10.12/Include/cpython/abstract.h:123
#111 call_function (kwnames=0x155554a2c7c0, oparg=<optimized out>, pp_stack=<synthetic pointer>, trace_info=0x7fffffff8fc0, tstate=<optimized out>) at python-3.10.12/Python/ceval.c:5893
#112 _PyEval_EvalFrameDefault (tstate=<optimized out>, f=0x5555624a1f80, throwflag=<optimized out>) at python-3.10.12/Python/ceval.c:4231
#113 0x000055555569899c in _PyEval_EvalFrame (throwflag=0, f=0x5555624a1f80, tstate=0x55555590a0c0) at python-3.10.12/Include/internal/pycore_ceval.h:46
#114 _PyEval_Vector (kwnames=<optimized out>, argcount=<optimized out>, args=<optimized out>, locals=0x0, con=0x1551de1d3b60, tstate=0x55555590a0c0) at python-3.10.12/Python/ceval.c:5067
#115 _PyFunction_Vectorcall (func=0x1551de1d3b50, stack=<optimized out>, nargsf=<optimized out>, kwnames=<optimized out>) at python-3.10.12/Objects/call.c:342
#116 0x0000555555688850 in _PyObject_VectorcallTstate (kwnames=0x0, nargsf=<optimized out>, args=0x155554b95ba8, callable=0x1551de1d3b50, tstate=0x55555590a0c0) at python-3.10.12/Include/cpython/abstract.h:114
#117 PyObject_Vectorcall (kwnames=0x0, nargsf=<optimized out>, args=0x155554b95ba8, callable=0x1551de1d3b50) at python-3.10.12/Include/cpython/abstract.h:123
#118 call_function (kwnames=0x0, oparg=<optimized out>, pp_stack=<synthetic pointer>, trace_info=0x7fffffff9170, tstate=<optimized out>) at python-3.10.12/Python/ceval.c:5893
#119 _PyEval_EvalFrameDefault (tstate=<optimized out>, f=0x155554b95a40, throwflag=<optimized out>) at python-3.10.12/Python/ceval.c:4213
#120 0x000055555572bf90 in _PyEval_EvalFrame (throwflag=0, f=0x155554b95a40, tstate=0x55555590a0c0) at python-3.10.12/Include/internal/pycore_ceval.h:46
#121 _PyEval_Vector (tstate=0x55555590a0c0, con=0x7fffffff9270, locals=<optimized out>, args=<optimized out>, argcount=<optimized out>, kwnames=<optimized out>) at python-3.10.12/Python/ceval.c:5067
#122 0x000055555572bed7 in PyEval_EvalCode (co=co@entry=0x155554a31dc0, globals=globals@entry=0x155554aabf80, locals=locals@entry=0x155554aabf80) at python-3.10.12/Python/ceval.c:1134
#123 0x000055555575c42a in run_eval_code_obj (tstate=0x55555590a0c0, co=0x155554a31dc0, globals=0x155554aabf80, locals=0x155554aabf80) at python-3.10.12/Python/pythonrun.c:1291
#124 0x0000555555757833 in run_mod (mod=<optimized out>, filename=<optimized out>, globals=0x155554aabf80, locals=0x155554aabf80, flags=<optimized out>, arena=<optimized out>) at python-3.10.12/Python/pythonrun.c:1312
#125 0x00005555555ee6cd in pyrun_file (fp=0x5555559348f0, filename=0x155554ab0490, start=<optimized out>, globals=0x155554aabf80, locals=0x155554aabf80, closeit=1, flags=0x7fffffff9458) at python-3.10.12/Python/pythonrun.c:1208
#126 0x0000555555751d1e in _PyRun_SimpleFileObject (fp=0x5555559348f0, filename=0x155554ab0490, closeit=1, flags=0x7fffffff9458) at python-3.10.12/Python/pythonrun.c:456
#127 0x00005555557518b4 in _PyRun_AnyFileObject (fp=0x5555559348f0, filename=0x155554ab0490, closeit=1, flags=0x7fffffff9458) at python-3.10.12/Python/pythonrun.c:90
#128 0x000055555574eaab in pymain_run_file_obj (skip_source_first_line=0, filename=0x155554ab0490, program_name=0x155554abb9b0) at python-3.10.12/Modules/main.c:357
#129 pymain_run_file (config=0x5555558ee130) at python-3.10.12/Modules/main.c:376
#130 pymain_run_python (exitcode=0x7fffffff9454) at python-3.10.12/Modules/main.c:591
#131 Py_RunMain () at python-3.10.12/Modules/main.c:670
#132 0x000055555571f527 in Py_BytesMain (argc=<optimized out>, argv=<optimized out>) at python-3.10.12/Modules/main.c:1090
#133 0x0000155555209d90 in __libc_start_call_main (main=main@entry=0x55555571f4e0 <main>, argc=argc@entry=10, argv=argv@entry=0x7fffffff9688) at ../sysdeps/nptl/libc_start_call_main.h:58
#134 0x0000155555209e40 in __libc_start_main_impl (main=0x55555571f4e0 <main>, argc=10, argv=0x7fffffff9688, init=<optimized out>, fini=<optimized out>, rtld_fini=<optimized out>, stack_end=0x7fffffff9678) at ../csu/libc-start.c:392
#135 0x000055555571f421 in _start ()