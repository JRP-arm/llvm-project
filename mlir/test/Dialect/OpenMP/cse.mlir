// RUN: mlir-opt %s -cse -split-input-file | FileCheck %s

// Recipe effects must invalidate loads even though the recipe is referenced
// by symbol and the single body has no effects.

omp.private {type = private} @recipe : memref<i32> init {
^bb0(%original: memref<i32>, %copy: memref<i32>):
  %zero = arith.constant 0 : i32
  memref.store %zero, %original[] : memref<i32>
  omp.yield(%copy : memref<i32>)
}

// CHECK-LABEL: func.func @init_writes_original(
// CHECK-SAME: %[[X:[^:]+]]: memref<i32>
func.func @init_writes_original(%x: memref<i32>) -> (i32, i32) {
  // CHECK: %[[BEFORE:.*]] = memref.load %[[X]][] : memref<i32>
  %before = memref.load %x[] : memref<i32>
  // CHECK: omp.single nowait
  omp.single nowait private(@recipe %x -> %copy : memref<i32>) {
    omp.terminator
  }
  // CHECK: %[[AFTER:.*]] = memref.load %[[X]][] : memref<i32>
  %after = memref.load %x[] : memref<i32>
  // CHECK: return %[[BEFORE]], %[[AFTER]] : i32, i32
  return %before, %after : i32, i32
}

// -----

// Recipe effects must invalidate loads even though the recipe is referenced
// by symbol and the single body has no effects.

omp.private {type = firstprivate} @recipe : memref<i32> copy {
^bb0(%original: memref<i32>, %copy: memref<i32>):
  %zero = arith.constant 0 : i32
  memref.store %zero, %original[] : memref<i32>
  omp.yield(%copy : memref<i32>)
}

// CHECK-LABEL: func.func @copy_writes_original(
// CHECK-SAME: %[[X:[^:]+]]: memref<i32>
func.func @copy_writes_original(%x: memref<i32>) -> (i32, i32) {
  // CHECK: %[[BEFORE:.*]] = memref.load %[[X]][] : memref<i32>
  %before = memref.load %x[] : memref<i32>
  // CHECK: omp.single nowait
  omp.single nowait private(@recipe %x -> %copy : memref<i32>) {
    omp.terminator
  }
  // CHECK: %[[AFTER:.*]] = memref.load %[[X]][] : memref<i32>
  %after = memref.load %x[] : memref<i32>
  // CHECK: return %[[BEFORE]], %[[AFTER]] : i32, i32
  return %before, %after : i32, i32
}

// -----

// Recipe effects must invalidate loads even though the recipe is referenced
// by symbol and the single body has no effects.
func.func private @unknown()
omp.private {type = private} @recipe : memref<i32> init {
^bb0(%original: memref<i32>, %copy: memref<i32>):
  func.call @unknown() : () -> ()
  omp.yield(%copy : memref<i32>)
}

// CHECK-LABEL: func.func @init_calls_unknown(
// CHECK-SAME: %[[X:[^:]+]]: memref<i32>
func.func @init_calls_unknown(%x: memref<i32>, %unrelated: memref<i32>) -> (i32, i32) {
  // CHECK: %[[BEFORE:.*]] = memref.load %[[X]][] : memref<i32>
  %before = memref.load %x[] : memref<i32>
  // CHECK: omp.single nowait
  omp.single nowait private(@recipe %unrelated -> %copy : memref<i32>) {
    omp.terminator
  }
  // CHECK: %[[AFTER:.*]] = memref.load %[[X]][] : memref<i32>
  %after = memref.load %x[] : memref<i32>
  // CHECK: return %[[BEFORE]], %[[AFTER]] : i32, i32
  return %before, %after : i32, i32
}

// -----

// Recipe effects must invalidate loads even though the recipe is referenced
// by symbol and the single body has no effects.
func.func private @unknown()
omp.private {type = private} @recipe : memref<i32> dealloc {
^bb0(%copy: memref<i32>):
  func.call @unknown() : () -> ()
  omp.yield
}

// CHECK-LABEL: func.func @dealloc_calls_unknown(
// CHECK-SAME: %[[X:[^:]+]]: memref<i32>
func.func @dealloc_calls_unknown(%x: memref<i32>, %unrelated: memref<i32>) -> (i32, i32) {
  // CHECK: %[[BEFORE:.*]] = memref.load %[[X]][] : memref<i32>
  %before = memref.load %x[] : memref<i32>
  // CHECK: omp.single nowait
  omp.single nowait private(@recipe %unrelated -> %copy : memref<i32>) {
    omp.terminator
  }
  // CHECK: %[[AFTER:.*]] = memref.load %[[X]][] : memref<i32>
  %after = memref.load %x[] : memref<i32>
  // CHECK: return %[[BEFORE]], %[[AFTER]] : i32, i32
  return %before, %after : i32, i32
}