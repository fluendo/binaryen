/*
 * Copyright 2025 WebAssembly Community Group participants
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

//
// FixFunctionBitcasts: Detect and fix call_indirect type mismatches.
//
// Scans all call_indirect instructions, traces their index operand back to a
// concrete table slot (through locals, stack stores/loads, and rodata), and
// for each mismatch (call site expects more params than the function accepts)
// generates a thunk with the expected signature, extends the table, and
// patches the slot reference to point to the thunk.
//
// Three patterns handled:
//   1. Direct local:  FP fp = cast(fn); fp(a,b);
//   2. Rodata struct: static Struct s = {cast(fn)}; s.fp(a,b);
//   3. Callback arg:  foo(cast(fn), a, b) where foo does fp(a,b,c)
//

#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "ir/find_all.h"
#include "pass.h"
#include "wasm-builder.h"
#include "wasm-traversal.h"
#include "wasm.h"

namespace wasm {

// ---------------------------------------------------------------------------
// ResolvedSlot: a constant table-slot index together with its patchable
// location in the IR.
//
// When we trace a call_indirect's index operand back to its origin we end up
// at one of two places:
//   - A Const* node inside the IR (stack-local or direct-constant pattern):
//     patching means updating that node's integer value in place.
//   - A 4-byte little-endian word inside a data segment (rodata struct
//     pattern): patching means overwriting those bytes.
// ---------------------------------------------------------------------------

struct ResolvedSlot {
  // The current table slot index this expression resolves to.
  uint32_t value = 0;

  // Set when the slot came from an IR Const node (local / immediate pattern).
  // Null when the slot came from a data segment.
  Const* constNode = nullptr;

  // Set when the slot came from a data segment (rodata pattern).
  struct { Index segIdx; uint32_t byteOff; } dataLoc = {0, 0};
  bool isData = false;

  // Overwrite the slot index at the origin location with newSlot.
  // Exactly one of the two branches will be taken depending on isData.
  void patch(uint32_t newSlot, Module* module) const {
    if (isData) {
      // Patch the 4-byte little-endian slot index stored in the data segment.
      auto& seg = module->dataSegments[dataLoc.segIdx];
      for (int i = 0; i < 4; i++)
        seg->data[dataLoc.byteOff + i] = (uint8_t)((newSlot >> (i * 8)) & 0xFF);
    } else if (constNode) {
      // Patch the integer value of the IR Const node directly.
      constNode->value = Literal((int32_t)newSlot);
    }
  }
};

// LocalDefs maps a local index to ALL expressions assigned to it within a
// function, built by a single post-order walk. Used to follow local.set →
// local.get chains during slot resolution. Keeping all assignments lets us
// find a constant in one branch even when other branches (e.g. asyncify
// shadow-stack restore) assign a non-constant to the same local.
using LocalDefs = std::unordered_map<Index, std::vector<Expression*>>;

// StoreMap maps (base-local-index, byte-offset) to the value expression last
// stored through that pointer+offset pair. Used to follow i32.store →
// i32.load chains when a function pointer is written into a stack-allocated
// struct and loaded back out.
using StoreMap  = std::unordered_map<std::pair<Index, uint32_t>, Expression*>;

// ---------------------------------------------------------------------------
// resolveToSlot
//
// Given the index operand of a call_indirect, attempts to trace it back to a
// concrete, constant table-slot index and returns a ResolvedSlot describing
// both the value and where to patch it if a fix is needed.
//
// Handles three source shapes:
//   Const         – the operand is a literal integer; the Const node itself
//                   is the patch site.
//   LocalGet      – the operand is a local; follow the last local.set
//                   assignment (via localDefs) and recurse.
//   Load i32      – the operand is a 4-byte load; two sub-cases:
//     ptr=Const   → rodata: walk data segments for a segment that covers the
//                   target address and read the slot from segment bytes.
//     ptr=LocalGet→ stack struct: look up the matching store in storeMap and
//                   recurse on the stored value.
//
// Returns nullopt when the operand cannot be traced to a constant.
// depth guards against runaway recursion on unusual IR.
// ---------------------------------------------------------------------------

static std::optional<ResolvedSlot>
resolveToSlot(Expression* expr, const LocalDefs& localDefs,
              const StoreMap& storeMap, Module* module, int depth = 0) {
  // Guard against deeply nested or cyclic chains.
  if (depth > 6) return std::nullopt;

  // Base case: the operand is already a literal integer constant.
  if (auto* c = expr->dynCast<Const>()) {
    return ResolvedSlot{(uint32_t)c->value.geti32(), c};
  }

  // LocalGet: try every assignment to this local and return the first one
  // that resolves to a constant. This handles asyncify-style functions where
  // one branch restores from the shadow stack (non-constant) and another
  // branch loads from a known data-section address (constant).
  if (auto* lg = expr->dynCast<LocalGet>()) {
    auto it = localDefs.find(lg->index);
    if (it != localDefs.end()) {
      for (Expression* def : it->second) {
        auto r = resolveToSlot(def, localDefs, storeMap, module, depth + 1);
        if (r) return r;
      }
    }
    return std::nullopt;
  }

  // Load: only consider 4-byte unsigned loads (i32.load / i32.load_u).
  if (auto* load = expr->dynCast<Load>()) {
    if (load->bytes == 4 && !load->signed_) {

      // Sub-case A: ptr is a constant address → rodata struct pattern.
      // The function pointer was placed in a data segment at link time.
      if (auto* ptrConst = load->ptr->dynCast<Const>()) {
        uint32_t addr = (uint32_t)ptrConst->value.geti32() + load->offset;
        // Walk every data segment looking for one whose range covers addr.
        for (Index si = 0; si < module->dataSegments.size(); ++si) {
          auto& seg = module->dataSegments[si];
          // Skip passive or non-const-offset segments.
          if (!seg->offset) continue;
          auto* c = seg->offset->dynCast<Const>();
          if (!c) continue;
          uint32_t base = (uint32_t)c->value.geti32();
          // Check that [addr, addr+4) lies within this segment's data.
          if (addr >= base && addr + 4 <= base + (uint32_t)seg->data.size()) {
            // Read the little-endian i32 from the segment bytes.
            uint32_t off = addr - base;
            uint32_t val = 0;
            for (int i = 0; i < 4; i++)
              val |= (uint8_t)seg->data[off + i] << (i * 8);
            ResolvedSlot rs;
            rs.value = val; rs.isData = true; rs.dataLoc = {si, off};
            return rs;
          }
        }
        return std::nullopt;
      }

      // Sub-case B: ptr is a local → stack-struct pattern.
      // The function pointer was stored into memory via a local pointer, then
      // loaded back out; follow the matching store in storeMap.
      if (auto* ptrLocal = load->ptr->dynCast<LocalGet>()) {
        auto key = std::make_pair(ptrLocal->index, (uint32_t)load->offset);
        auto it = storeMap.find(key);
        if (it != storeMap.end())
          return resolveToSlot(it->second, localDefs, storeMap, module, depth + 1);
      }
    }
  }
  return std::nullopt;
}

// ---------------------------------------------------------------------------
// traceToParam
//
// Given an expression that appears as the index operand of a call_indirect,
// attempts to trace it back to a parameter of `func`. This is used for the
// interprocedural (callback-argument) pattern: callee receives a function
// pointer through an argument and calls it indirectly.
//
// Returns the parameter index if the expression ultimately comes from a
// parameter, or nullopt if it cannot be resolved to one.
//
// Follows the same local.get → local.set chain as resolveToSlot, and also
// follows i32.load → i32.store chains for pointer-passed structs.
// depth guards against runaway recursion.
// ---------------------------------------------------------------------------

static std::optional<Index>
traceToParam(Expression* expr, const LocalDefs& localDefs,
             const StoreMap& storeMap, Function* func, int depth = 0) {
  // Guard against deeply nested or cyclic chains.
  if (depth > 6) return std::nullopt;

  // LocalGet: if this local index is in the parameter range, we're done.
  // Otherwise try every assignment and return the first that traces to a param.
  if (auto* lg = expr->dynCast<LocalGet>()) {
    if (lg->index < func->getNumParams()) return lg->index;
    auto it = localDefs.find(lg->index);
    if (it != localDefs.end()) {
      for (Expression* def : it->second) {
        auto r = traceToParam(def, localDefs, storeMap, func, depth + 1);
        if (r) return r;
      }
    }
    return std::nullopt;
  }

  // Load: only consider 4-byte unsigned loads; follow the matching store.
  if (auto* load = expr->dynCast<Load>()) {
    if (load->bytes == 4 && !load->signed_) {
      // The pointer base must be a local so we can look it up in storeMap.
      if (auto* ptrLocal = load->ptr->dynCast<LocalGet>()) {
        auto key = std::make_pair(ptrLocal->index, (uint32_t)load->offset);
        auto it = storeMap.find(key);
        if (it != storeMap.end())
          return traceToParam(it->second, localDefs, storeMap, func, depth + 1);
      }
    }
  }
  return std::nullopt;
}

// buildLocalDefs: walk func's body and record the RHS of every local.set.
// When a local is set more than once, the last assignment wins (sufficient
// for the patterns we handle, which always use a single static slot value).
static LocalDefs buildLocalDefs(Function* func) {
  LocalDefs defs;
  struct C : PostWalker<C> {
    LocalDefs& d;
    C(LocalDefs& d) : d(d) {}
    void visitLocalSet(LocalSet* s) { d[s->index].push_back(s->value); }
  } c(defs);
  c.walk(func->body);
  return defs;
}

// buildStoreMap: walk func's body and record the value of every 4-byte store
// keyed by (base-local-index, byte-offset). Used to find what was written
// into a stack-allocated struct field that is later loaded back.
static StoreMap buildStoreMap(Function* func) {
  StoreMap stores;
  struct C : PostWalker<C> {
    StoreMap& s;
    C(StoreMap& s) : s(s) {}
    void visitStore(Store* curr) {
      // Only track 4-byte (i32) stores whose base pointer is a local.
      if (curr->bytes != 4) return;
      if (auto* base = curr->ptr->dynCast<LocalGet>())
        s[{base->index, (uint32_t)curr->offset}] = curr->value;
    }
  } c(stores);
  c.walk(func->body);
  return stores;
}

// ---------------------------------------------------------------------------
// Main pass
// ---------------------------------------------------------------------------

struct FixFunctionBitcasts : public Pass {
  // Maps table-name → (slot-index → function-name) for every active element
  // segment in the module. Built once in buildTableEntries() before analysis.
  std::unordered_map<Name, std::unordered_map<uint32_t, Name>> tableEntries;

  // Maps function-name → list of (calling-function, Call* node) for every
  // direct Call instruction that targets that function. Used for the
  // interprocedural (callback-argument) pattern to trace backwards from a
  // callee to its call sites. Built once in buildCallerMap().
  std::unordered_map<Name, std::vector<std::pair<Function*, Call*>>> callerMap;

  // All the information needed to apply a single fix: which slot to redirect,
  // what thunk signature to generate, and diagnostic context.
  struct PendingFix {
    ResolvedSlot ref;          // The patchable slot reference found during analysis.
    uint32_t originalSlot;     // Original slot value (for diagnostics).
    HeapType callType;         // The call_indirect's expected heap type (wider sig).
    Name calledFuncName;       // The function currently at that slot (narrower sig).
    Name tableName;            // The table involved.
    Name contextFunc;          // The function containing the call_indirect.
    std::optional<Name> callerFunc; // Set for interprocedural fixes: the outer
                                    // function that passes the slot as an argument.
  };

  // Fixes collected during the analysis phase, applied all at once afterward
  // to avoid invalidating iterators.
  std::vector<PendingFix> pendingFixes;

  // Deduplication cache: thunk-name → new table slot. Prevents generating
  // the same thunk twice when multiple call sites share the same slot.
  std::unordered_map<std::string, uint32_t> thunkCache;

  // buildTableEntries: populate tableEntries from all active (non-passive)
  // element segments that have a constant offset. Only RefFunc entries are
  // recorded; other element kinds (e.g. ref.null) are ignored.
  void buildTableEntries(Module* module) {
    for (auto& seg : module->elementSegments) {
      // Skip passive segments (no associated table) and declarative segments.
      if (!seg->table.is()) continue;
      // Skip segments with non-constant or absent offsets.
      auto* c = seg->offset ? seg->offset->dynCast<Const>() : nullptr;
      if (!c) continue;
      uint32_t base = (uint32_t)c->value.geti32();
      // Record each ref.func entry at its absolute slot index.
      for (Index i = 0; i < seg->data.size(); ++i)
        if (auto* ref = seg->data[i]->dynCast<RefFunc>())
          tableEntries[seg->table][base + i] = ref->func;
    }
  }

  // buildCallerMap: populate callerMap by walking every function and collecting
  // all direct Call instructions. This enables the interprocedural pattern:
  // given a callee function that uses a parameter as a call_indirect index, we
  // can find all call sites to check what value was passed for that parameter.
  void buildCallerMap(Module* module) {
    for (auto& func : module->functions) {
      // Skip imported functions — they have no body to walk.
      if (func->imported()) continue;
      FindAll<Call> calls(func->body);
      for (auto* call : calls.list)
        callerMap[call->target].emplace_back(func.get(), call);
    }
  }

  // run: entry point for the pass. Runs to fixpoint: each iteration may
  // create new thunks at new table slots, and those new slots can themselves
  // appear in call chains that weren't resolvable in earlier iterations.
  // We rebuild tableEntries each round (since applyFixes adds element
  // segments) and stop when no new mismatches are found.  thunkCache persists
  // across iterations so we never generate the same thunk twice.
  void run(Module* module) override {
    buildCallerMap(module);
    // Safety limit: in practice fixpoint is reached in ≤3 iterations.
    for (int iter = 0; iter < 10; ++iter) {
      tableEntries.clear();
      buildTableEntries(module);
      pendingFixes.clear();
      for (auto& func : module->functions) {
        if (func->imported()) continue;
        detectInFunction(func.get(), module);
      }
      if (pendingFixes.empty()) break;
      applyFixes(module);
    }
  }

  // detectInFunction: scan every call_indirect in func and try to resolve its
  // index operand to a concrete table slot. If successful, and if the target
  // function's actual signature is narrower than the call site's expected
  // signature, queue a fix. Two resolution strategies are tried in order:
  //
  //   1. Intra-procedural: resolveToSlot directly traces the index operand
  //      back to a constant within this function (local variable or rodata).
  //
  //   2. Interprocedural: traceToParam finds that the index came from a
  //      parameter, then resolveToSlot is called on each call site's argument
  //      for that parameter across all known callers.
  void detectInFunction(Function* func, Module* module) {
    auto localDefs = buildLocalDefs(func);
    auto storeMap  = buildStoreMap(func);
    FindAll<CallIndirect> calls(func->body);

    for (auto* call : calls.list) {
      // Only examine call_indirects that use a table we have entries for.
      auto tableIt = tableEntries.find(call->table);
      if (tableIt == tableEntries.end()) continue;

      // Strategy 1: try to resolve the index directly within this function.
      auto slotOpt = resolveToSlot(call->target, localDefs, storeMap, module);
      if (slotOpt) {
        queueIfMismatch(*slotOpt, tableIt->second, call->heapType,
                        call->table, func->name, module);
        continue;
      }

      // Strategy 2: the index may come from a parameter (callback pattern).
      // Determine which parameter carries the slot value.
      auto paramOpt = traceToParam(call->target, localDefs, storeMap, func);
      if (!paramOpt) continue;
      Index paramIdx = *paramOpt;

      // Find all direct callers of this function and resolve the argument they
      // pass for paramIdx.
      auto callerIt = callerMap.find(func->name);
      if (callerIt == callerMap.end()) continue;
      for (auto& [callerFunc, callExpr] : callerIt->second) {
        // Skip imported callers — they have no body to build defs/stores from.
        if (callerFunc->imported()) continue;
        // Guard against calls that don't supply enough arguments.
        if (paramIdx >= callExpr->operands.size()) continue;
        auto callerLocalDefs = buildLocalDefs(callerFunc);
        auto callerStoreMap  = buildStoreMap(callerFunc);
        auto callerSlot = resolveToSlot(callExpr->operands[paramIdx],
                                        callerLocalDefs, callerStoreMap, module);
        if (!callerSlot) continue;
        queueIfMismatch(*callerSlot, tableIt->second, call->heapType,
                        call->table, func->name, module, callerFunc->name);
      }
    }
  }

  // queueIfMismatch: given a resolved slot and the call_indirect's expected
  // heap type, check whether the function actually at that slot has a
  // different (narrower) signature. If so, add a PendingFix to pendingFixes.
  //
  // Only fixes where the call site passes *more* arguments than the target
  // accepts are handled — the thunk simply discards the extra arguments.
  // Mismatches in the other direction (too few arguments) are not fixable
  // without fabricating argument values and are left alone.
  void queueIfMismatch(const ResolvedSlot& ref,
                       const std::unordered_map<uint32_t, Name>& slots,
                       HeapType callType, Name tableName, Name contextFunc,
                       Module* module,
                       std::optional<Name> callerFunc = std::nullopt) {
    // Check that the slot is occupied by a known function.
    auto slotIt = slots.find(ref.value);
    if (slotIt == slots.end()) return;
    auto* targetFn = module->getFunctionOrNull(slotIt->second);
    if (!targetFn) return;

    // Compare the actual function type with the type the call_indirect expects.
    HeapType actualType = targetFn->type.getHeapType();
    if (callType == actualType) return; // Types match — nothing to do.

    Signature callSig   = callType.getSignature();
    Signature actualSig = actualType.getSignature();

    // Only handle the case where the call site passes more arguments than the
    // function accepts; the excess arguments will be dropped by the thunk.
    if (callSig.params.size() <= actualSig.params.size()) return;

    pendingFixes.push_back(
      {ref, ref.value, callType, slotIt->second, tableName,
       contextFunc, callerFunc});
  }

  // applyFixes: iterate over all queued fixes, generate (or reuse) a thunk for
  // each, patch the slot reference to point at the thunk, and print a
  // diagnostic line to stdout.
  void applyFixes(Module* module) {
    for (auto& fix : pendingFixes) {
      uint32_t newSlot = getOrCreateThunk(fix, module);
      // UINT32_MAX is the sentinel value meaning the fix was skipped due to
      // irreconcilable parameter type mismatch.
      if (newSlot == UINT32_MAX) continue;
      // Rewrite the slot reference in the IR or data segment.
      fix.ref.patch(newSlot, module);

      // Collect signature info for the diagnostic message.
      auto* origFn    = module->getFunctionOrNull(fix.calledFuncName);
      HeapType actual = origFn->type.getHeapType();
      Signature callSig   = fix.callType.getSignature();
      Signature actualSig = actual.getSignature();

      std::cout << "[fix-function-bitcasts] FIXED";
      // For interprocedural fixes, show the caller → callee chain and the
      // originating slot so users can locate the source of the mismatch.
      if (fix.callerFunc)
        std::cout << " (via `" << *fix.callerFunc << "` \xe2\x86\x92 `"
                  << fix.contextFunc << "`, param carries slot "
                  << fix.originalSlot << ")";
      else
        std::cout << " in `" << fix.contextFunc << "`";
      std::cout << ":\n"
                << "  `" << fix.calledFuncName << "` " << actual
                << "\n  called as " << fix.callType
                << "\n  \xe2\x86\x92 thunk at slot " << newSlot
                << " (" << (callSig.params.size() - actualSig.params.size())
                << " extra arg(s) dropped)\n";
    }
  }

  // getOrCreateThunk: return the table slot of a thunk that bridges the gap
  // between the call site's expected signature (fix.callType, wider) and the
  // target function's actual signature (narrower).
  //
  // The thunk has the wider signature so call_indirect succeeds type-checking,
  // then immediately calls the original function forwarding only the arguments
  // it accepts; extra arguments are simply not passed (dropped).
  //
  // If a thunk for this (target, call-arity) combination was already created
  // during this pass invocation, the cached slot is returned instead of
  // generating a duplicate.
  uint32_t getOrCreateThunk(const PendingFix& fix, Module* module) {
    auto* origFn    = module->getFunction(fix.calledFuncName);
    Signature callSig   = fix.callType.getSignature();
    Signature actualSig = origFn->type.getHeapType().getSignature();

    // Thunk name encodes the target and the call-site arity to be unique per
    // (function, arity) pair.
    std::string thunkName = "byn$fpfix$" + fix.calledFuncName.toString()
                            + "$" + std::to_string(callSig.params.size());

    // Return the cached slot if this thunk was already generated.
    auto it = thunkCache.find(thunkName);
    if (it != thunkCache.end()) return it->second;

    // When the pass is run more than once on the same module (e.g. via
    // multiple --fix-function-bitcasts flags), thunkCache is empty for the
    // second invocation but the thunk function and its table slot already
    // exist.  Detect that situation and return the existing slot rather than
    // trying to re-add the function.
    if (module->getFunctionOrNull(Name(thunkName))) {
      auto tableIt = tableEntries.find(fix.tableName);
      if (tableIt != tableEntries.end()) {
        for (auto& [slot, fname] : tableIt->second) {
          if (fname == Name(thunkName)) {
            thunkCache[thunkName] = slot;
            return slot;
          }
        }
      }
      // Function exists but slot not in tableEntries (shouldn't happen).
      // Treat as already-fixed and skip.
      thunkCache[thunkName] = UINT32_MAX;
      return UINT32_MAX;
    }

    Builder builder(*module);

    // Build the thunk function body: forward only the arguments the original
    // function accepts (indices 0..actualSig.params.size()-1), ignoring the
    // extra arguments that the call site passes.
    auto thunkFunc = Builder::makeFunction(Name(thunkName), fix.callType, {});
    // The thunk's parameter locals have the types from callSig (the wider
    // signature). We forward only the first actualSig.params.size() of them.
    // Use callSig.params[i] for the local.get type — that is the actual type
    // of local i in the thunk. If it differs from actualSig.params[i] we
    // skip forwarding that argument (treat it as an irreconcilable mismatch).
    std::vector<Expression*> args;
    bool typeMismatch = false;
    for (Index i = 0; i < actualSig.params.size(); ++i) {
      if (callSig.params[i] != actualSig.params[i]) {
        typeMismatch = true;
        break;
      }
      args.push_back(builder.makeLocalGet(i, callSig.params[i]));
    }
    if (typeMismatch) {
      // Cannot safely forward — generate an unreachable body and skip.
      thunkFunc->body = builder.makeUnreachable();
      module->addFunction(std::move(thunkFunc));
      thunkCache[thunkName] = UINT32_MAX; // sentinel: no valid slot
      return UINT32_MAX;
    }
    Expression* body =
      builder.makeCall(fix.calledFuncName, args, actualSig.results);

    if (actualSig.results != callSig.results) {
      if (callSig.results == Type::none) {
        // Call site expects void but target returns a value: drop it.
        body = builder.makeDrop(body);
      } else if (actualSig.results == Type::none && callSig.results.isConcrete()) {
        // Target returns void but call site expects a value: call then
        // synthesize a zero of the expected type. Only handles the common
        // single-value case (i32/i64/f32/f64); tuples and reference types are
        // left as unreachable since we cannot fabricate a valid value.
        Type retType = callSig.results;
        Expression* zero;
        if (retType == Type::i32)      zero = builder.makeConst(Literal(int32_t(0)));
        else if (retType == Type::i64) zero = builder.makeConst(Literal(int64_t(0)));
        else if (retType == Type::f32) zero = builder.makeConst(Literal(float(0.0f)));
        else if (retType == Type::f64) zero = builder.makeConst(Literal(double(0.0)));
        else                           zero = builder.makeUnreachable();
        body = builder.makeSequence(body, zero, retType);
      }
    }
    thunkFunc->body = body;
    module->addFunction(std::move(thunkFunc));

    // Extend the table by one slot to hold the new thunk.
    auto* table  = module->getTable(fix.tableName);
    uint32_t newSlot = table->initial;
    table->initial++;
    // If the table has a declared maximum, grow it too.
    if (table->max != Table::kUnlimitedSize) table->max++;

    // Retrieve the added function to obtain its exact declared type, which is
    // required for the ref.func expression in the element segment.
    auto* addedThunk = module->getFunction(Name(thunkName));

    // Create an active element segment that initialises the new slot.
    auto seg    = std::make_unique<ElementSegment>();
    seg->name   = Name("fpfix$elem$" + std::to_string(newSlot));
    seg->table  = fix.tableName;
    seg->type   = Type(HeapType::func, Nullable);
    seg->offset = builder.makeConst(Literal(int32_t(newSlot)));
    // addedThunk->type carries the Exact flag required by the wasm validator
    // for ref.func expressions referencing defined (non-imported) functions.
    seg->data.push_back(builder.makeRefFunc(Name(thunkName), addedThunk->type));
    module->addElementSegment(std::move(seg));

    thunkCache[thunkName] = newSlot;
    return newSlot;
  }
};

Pass* createFixFunctionBitcastsPass() { return new FixFunctionBitcasts(); }

} // namespace wasm
