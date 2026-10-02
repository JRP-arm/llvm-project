// RUN: mlir-opt %s --test-side-effects --verify-diagnostics

func.func @threadprivate_effects(%sym_addr: memref<i32>) {
  // expected-remark@+5 {{found an instance of 'read' on op operand 0, on resource '<Default>'}}
  // expected-remark@+4 {{found an instance of 'write' on op result 0, on resource '<Default>'}}
  // expected-remark@+3 {{found an instance of 'allocate' on resource '<Default>'}}
  // expected-remark@+2 {{found an instance of 'read' on resource 'OpenMPThreadprivateRuntime'}}
  // expected-remark@+1 {{found an instance of 'write' on resource 'OpenMPThreadprivateRuntime'}}
  %tls_addr = omp.threadprivate %sym_addr
      : memref<i32> -> memref<i32>
  return
}
