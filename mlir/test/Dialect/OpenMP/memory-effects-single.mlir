// RUN: mlir-opt %s --test-side-effects --verify-diagnostics


omp.private {type = private} @p : memref<i32>
func.func @single_private(%x: memref<i32>) {
  // expected-remark@+3 {{found an instance of 'allocate' on block argument 0, on resource '<Default>'}}
  // expected-remark@+2 {{found an instance of 'free' on block argument 0, on resource '<Default>'}}
  // expected-remark@+1 {{found an instance of 'read' on op operand 0, on resource '<Default>'}}
  omp.single nowait private(@p %x -> %private_x : memref<i32>) {
    omp.terminator
  }
  return
}

omp.private {type = firstprivate} @fp : memref<i32> copy {
^bb0(%original: memref<i32>, %copy: memref<i32>):
  // expected-remark@+1 {{found an instance of 'read' on op operand 0, on resource '<Default>'}}
  %v = memref.load %original[] : memref<i32>
  // expected-remark@+1 {{found an instance of 'write' on op operand 1, on resource '<Default>'}}
  memref.store %v, %copy[] : memref<i32>
  omp.yield(%copy : memref<i32>)
}

func.func @single_firstprivate(%x: memref<i32>) {
  // expected-remark@+5 {{found an instance of 'allocate' on block argument 0, on resource '<Default>'}}
  // expected-remark@+4 {{found an instance of 'free' on block argument 0, on resource '<Default>'}}
  // expected-remark@+3 {{found an instance of 'read' on op operand 0, on resource '<Default>'}}
  // expected-remark@+2 {{found an instance of 'read' on resource '<Default>'}}
  // expected-remark@+1 {{found an instance of 'write' on resource '<Default>'}}
  omp.single nowait private(@fp %x -> %private_x : memref<i32>) {
    omp.terminator
  }
  return
}
