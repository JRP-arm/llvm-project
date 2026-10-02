// RUN: mlir-opt %s --test-side-effects --verify-diagnostics

omp.declare_reduction @add_f32 : f32
init {
^bb0(%unused: f32):
  // expected-remark@+1 {{operation has no memory effects}}
  %zero = arith.constant 0.0 : f32
  omp.yield(%zero : f32)
}
combiner {
^bb0(%lhs: f32, %rhs: f32):
  // expected-remark@+1 {{operation has no memory effects}}
  %sum = arith.addf %lhs, %rhs : f32
  omp.yield(%sum : f32)
}

func.func @taskgroup_reduction(%x: memref<f32>) {
  // expected-remark@+4 {{found an instance of 'allocate' on block argument 0, on resource '<Default>'}}
  // expected-remark@+3 {{found an instance of 'free' on block argument 0, on resource '<Default>'}}
  // expected-remark@+2 {{found an instance of 'read' on resource '<Default>'}}
  // expected-remark@+1 {{found an instance of 'write' on resource '<Default>'}}
  omp.taskgroup task_reduction(
      @add_f32 %x -> %private_x : memref<f32>) {
    omp.terminator
  }
  return
}

func.func @taskwait_plain() {
  // expected-remark@+2 {{found an instance of 'read' on resource '<Default>'}}
  // expected-remark@+1 {{found an instance of 'write' on resource '<Default>'}}
  omp.taskwait
  return
}

func.func @taskwait_nowait_depend(%x: memref<i32>) {
  // expected-remark@+2 {{found an instance of 'read' on resource '<Default>'}}
  // expected-remark@+1 {{found an instance of 'write' on resource '<Default>'}}
  omp.taskwait depend(taskdependout -> %x : memref<i32>) nowait
  return
}

func.func @taskwait_depend_iterated(%ptr : !omp.iterated<!llvm.ptr>) {
  // expected-remark@+2 {{found an instance of 'read' on resource '<Default>'}}
  // expected-remark@+1 {{found an instance of 'write' on resource '<Default>'}}
  omp.taskwait depend(taskdependout -> %ptr : !omp.iterated<!llvm.ptr>) nowait
  return
}
