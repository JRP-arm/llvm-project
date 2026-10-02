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

func.func @sections_effects() {
  // An implicit barrier reads and writes unspecified memory.
  // expected-remark@+2 {{found an instance of 'read' on resource '<Default>'}}
  // expected-remark@+1 {{found an instance of 'write' on resource '<Default>'}}
  omp.sections {
    omp.section {
      omp.terminator
    }
    omp.terminator
  }
  return
}

func.func @sections_nowait_no_effects() {
  // expected-remark@+1 {{operation has no memory effects}}
  omp.sections nowait {
    omp.section {
      omp.terminator
    }
    omp.terminator
  }
  return
}

omp.private {type = private} @sections_p : memref<i32>

func.func @sections_private_effects(%x: memref<i32>) {
  // expected-remark@+5 {{found an instance of 'allocate' on block argument 0, on resource '<Default>'}}
  // expected-remark@+4 {{found an instance of 'free' on block argument 0, on resource '<Default>'}}
  // expected-remark@+3 {{found an instance of 'read' on op operand 0, on resource '<Default>'}}
  // expected-remark@+2 {{found an instance of 'read' on block argument 0, on resource '<Default>'}}
  // expected-remark@+1 {{found an instance of 'write' on op operand 0, on resource '<Default>'}}
  omp.sections nowait private(@sections_p %x -> %private_x : memref<i32>) {
    omp.section {
    ^bb0(%section_x: memref<i32>):
      omp.terminator
    }
    omp.terminator
  }
  return
}

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

func.func @sections_reduction_effects(%x: memref<f32>) {
  // TODO: Nowait does not eliminate an implicit barrier if the reduction clause is also present,
  // requiring the below two remarks
  // expected-remark@+7 {{found an instance of 'read' on resource '<Default>'}}
  // expected-remark@+6 {{found an instance of 'write' on resource '<Default>'}}
  // expected-remark@+5 {{found an instance of 'allocate' on block argument 0, on resource '<Default>'}}
  // expected-remark@+4 {{found an instance of 'free' on block argument 0, on resource '<Default>'}}
  // expected-remark@+3 {{found an instance of 'read' on block argument 0, on resource '<Default>'}}
  // expected-remark@+2 {{found an instance of 'read' on op operand 0, on resource '<Default>'}}
  // expected-remark@+1 {{found an instance of 'write' on op operand 0, on resource '<Default>'}}
  omp.sections nowait reduction(
      @add_f32 %x -> %private_x : memref<f32>) {
    omp.section {
    ^bb0(%section_x: memref<f32>):
      omp.terminator
    }
    omp.terminator
  }
  return
}

func.func @critical_effects() {
  // expected-remark@+2 {{found an instance of 'read' on resource '<Default>'}}
  // expected-remark@+1 {{found an instance of 'write' on resource '<Default>'}}
  omp.critical {
    omp.terminator
  }
  return
}

