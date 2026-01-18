; Tree-sitter folding queries for Simplex
; Used by editors to determine which regions can be folded

; =============================================================================
; Definitions
; =============================================================================

; Function bodies
(function_definition
  body: (block) @fold)

; Struct bodies
(struct_definition
  (struct_body) @fold)

; Enum bodies
(enum_definition
  (enum_body) @fold)

; Trait bodies
(trait_definition
  (trait_body) @fold)

; Impl bodies
(impl_block
  (impl_body) @fold)

; Module bodies
(mod_declaration
  (mod_body) @fold)

; Extern blocks
(extern_block
  (extern_body) @fold)

; Actor bodies
(actor_definition
  (actor_body) @fold)

; Receive handler bodies
(receive_handler
  (block) @fold)

; Specialist bodies
(specialist_definition
  (specialist_body) @fold)

; Hive bodies
(hive_definition
  (hive_body) @fold)

; Anima bodies
(anima_definition
  (anima_body) @fold)

; =============================================================================
; Control Flow
; =============================================================================

; Block expressions
(block) @fold

; If expression branches
(if_expression
  consequence: (block) @fold)

(if_expression
  alternative: (block) @fold)

; Match expression body
(match_expression) @fold

; For loop bodies
(for_expression
  body: (block) @fold)

; While loop bodies
(while_expression
  body: (block) @fold)

; Loop bodies
(loop_expression
  body: (block) @fold)

; Closure bodies
(closure_expression
  (block) @fold)

; =============================================================================
; Containers
; =============================================================================

; Array expressions with multiple elements
(array_expression) @fold

; Struct expressions
(struct_expression) @fold

; Use groups
(use_group) @fold

; Parameters (when spanning multiple lines)
(parameters) @fold

; Arguments (when spanning multiple lines)
(arguments) @fold

; Generic parameters
(generic_parameters) @fold

; Type arguments
(type_arguments) @fold

; =============================================================================
; Comments
; =============================================================================

; Multi-line block comments
(block_comment) @fold

; Consecutive line comments (handled by editor)
(line_comment) @fold
