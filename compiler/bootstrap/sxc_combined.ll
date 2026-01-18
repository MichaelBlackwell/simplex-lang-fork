; ModuleID = 'simplex_program'
target triple = "x86_64-apple-macosx14.0.0"

declare ptr @malloc(i64)
declare void @free(ptr)
declare void @intrinsic_println(ptr)
declare void @intrinsic_print(ptr)
declare void @"print_i64"(i64)
declare ptr @intrinsic_int_to_string(i64)
declare ptr @intrinsic_string_new(ptr)
declare ptr @intrinsic_string_from_char(i64)
declare i64 @intrinsic_string_len(ptr)
declare ptr @intrinsic_string_concat(ptr, ptr)
declare ptr @intrinsic_string_slice(ptr, i64, i64)
declare i64 @intrinsic_string_char_at(ptr, i64)
declare i1 @intrinsic_string_eq(ptr, ptr)
declare i64 @intrinsic_string_to_int(ptr)
declare ptr @intrinsic_vec_new()
declare void @intrinsic_vec_push(ptr, ptr)
declare ptr @intrinsic_vec_get(ptr, i64)
declare i64 @intrinsic_vec_len(ptr)
declare void @intrinsic_vec_set(ptr, i64, ptr)
declare ptr @intrinsic_vec_pop(ptr)
declare void @intrinsic_vec_clear(ptr)
declare void @intrinsic_vec_remove(ptr, i64)
declare ptr @intrinsic_get_args()
declare ptr @intrinsic_read_file(ptr)
declare void @intrinsic_write_file(ptr, ptr)
declare ptr @store_ptr(ptr, i64, ptr)
declare ptr @store_i64(ptr, i64, i64)
declare ptr @load_ptr(ptr, i64)
declare i64 @load_i64(ptr, i64)
; Timing intrinsics
declare i64 @intrinsic_get_time_ms()
declare i64 @intrinsic_get_time_us()
; Arena allocator intrinsics
declare ptr @intrinsic_arena_create(i64)
declare ptr @intrinsic_arena_alloc(ptr, i64)
declare void @intrinsic_arena_reset(ptr)
declare void @intrinsic_arena_free(ptr)
declare i64 @intrinsic_arena_used(ptr)
; StringBuilder intrinsics
declare ptr @intrinsic_sb_new()
declare ptr @intrinsic_sb_new_cap(i64)
declare void @intrinsic_sb_append(ptr, ptr)
declare void @intrinsic_sb_append_char(ptr, i64)
declare void @intrinsic_sb_append_i64(ptr, i64)
declare ptr @intrinsic_sb_to_string(ptr)
declare void @intrinsic_sb_clear(ptr)
declare void @intrinsic_sb_free(ptr)
declare i64 @intrinsic_sb_len(ptr)
; File I/O intrinsics
declare ptr @intrinsic_getenv(ptr)
declare i64 @intrinsic_file_exists(ptr)
declare i64 @intrinsic_is_file(ptr)
declare i64 @intrinsic_is_directory(ptr)
declare i64 @intrinsic_file_size(ptr)
declare i64 @intrinsic_file_mtime(ptr)
declare i64 @intrinsic_remove_path(ptr)
declare i64 @intrinsic_mkdir_p(ptr)
declare ptr @intrinsic_get_cwd()
declare i64 @intrinsic_set_cwd(ptr)
declare ptr @intrinsic_list_dir(ptr)
declare ptr @intrinsic_path_join(ptr, ptr)
declare ptr @intrinsic_path_dirname(ptr)
declare ptr @intrinsic_path_basename(ptr)
declare ptr @intrinsic_path_extension(ptr)
declare i64 @file_copy(ptr, ptr)
declare i64 @file_rename(ptr, ptr)
declare void @stderr_write(ptr)
declare void @stderr_writeln(ptr)
; Error handling intrinsics
declare void @intrinsic_panic(ptr)
declare void @intrinsic_print_stack_trace()
; Process intrinsics
declare i64 @intrinsic_process_run(ptr)
declare ptr @intrinsic_process_output(ptr)
; Phase 20: REPL/I/O intrinsics
declare ptr @intrinsic_read_line()
declare i64 @intrinsic_is_tty()
declare i64 @intrinsic_stdin_has_data()
declare i64 @intrinsic_string_hash(ptr)
declare i64 @intrinsic_string_find(ptr, ptr, i64)
declare ptr @intrinsic_string_trim(ptr)
declare ptr @intrinsic_string_split(ptr, ptr)
declare i64 @intrinsic_string_starts_with(ptr, ptr)
declare i64 @intrinsic_string_ends_with(ptr, ptr)
declare i64 @intrinsic_string_contains(ptr, ptr)
declare ptr @intrinsic_string_replace(ptr, ptr, ptr)
declare ptr @intrinsic_string_lines(ptr)
declare ptr @intrinsic_string_join(ptr, ptr)
declare ptr @intrinsic_string_to_lowercase(ptr)
declare ptr @intrinsic_string_to_uppercase(ptr)
declare i64 @intrinsic_string_compare(ptr, ptr)
declare i64 @intrinsic_copy_file(ptr, ptr)
declare ptr @intrinsic_get_home_dir()
declare i64 @cli_getenv(i64)
declare i64 @file_read(i64)
declare void @file_write(i64, i64)
declare i64 @remove_path(i64)
declare i64 @f64_parse(ptr)
; Generator intrinsics
declare i64 @generator_yield(i64)
declare i64 @generator_new(i64)
declare i64 @generator_next(i64)
declare i64 @generator_done(i64)
; Belief guard intrinsics
declare i64 @belief_guard_get_confidence(i64)
declare i64 @belief_guard_get_derivative(i64)
declare i64 @belief_register(i64, double)
declare void @belief_update(i64, double)
declare double @belief_get_value(i64)
declare i64 @future_poll(i64)
declare i64 @future_ready(i64)
declare i64 @future_pending()
declare void @executor_run(i64)
declare i64 @executor_spawn(i64)
declare i64 @async_join(i64, i64)
declare i64 @join_result1(i64)
declare i64 @join_result2(i64)
declare i64 @async_select(i64, i64)
declare i64 @select_result(i64)
declare i64 @select_which(i64)
declare i64 @async_timeout(i64, i64)
declare i64 @timeout_result(i64)
declare i64 @timeout_expired(i64)
declare i64 @time_now_ms()
declare i64 @pin_new(i64, i64)
declare i64 @pin_new_uninit(i64)
declare i64 @pin_get(i64)
declare i64 @pin_get_mut(i64)
declare i64 @pin_is_pinned(i64)
declare void @pin_ref(i64)
declare void @pin_unref(i64)
declare void @pin_set_self_ref(i64, i64)
declare i64 @pin_check_self_ref(i64, i64)
declare i64 @intrinsic_call0(i64)
declare i64 @intrinsic_call1(i64, i64)
declare i64 @intrinsic_call2(i64, i64, i64)
declare i64 @intrinsic_call3(i64, i64, i64, i64)
declare i64 @scope_new()
declare i64 @scope_spawn(i64, i64)
declare i64 @scope_poll(i64)
declare i64 @scope_join(i64)
declare i64 @scope_get_result(i64, i64)
declare void @scope_cancel(i64)
declare i64 @scope_count(i64)
declare i64 @scope_completed(i64)
declare void @scope_free(i64)
declare i64 @nursery_run(i64, i64)
declare i64 @actor_get_status(i64)
declare i64 @actor_get_exit_reason(i64)
declare i64 @actor_get_error_code(i64)
declare void @actor_set_error(i64, i64, i64)
declare void @actor_stop(i64)
declare void @actor_kill(i64)
declare void @actor_crash(i64, i64, i64)
declare void @actor_set_on_error(i64, i64)
declare void @actor_set_on_exit(i64, i64)
declare void @actor_set_supervisor(i64, i64)
declare i64 @actor_get_supervisor(i64)
declare i64 @actor_get_restart_count(i64)
declare void @actor_increment_restart(i64)
declare i64 @actor_is_alive(i64)
declare i64 @circuit_breaker_new(i64, i64, i64)
declare i64 @circuit_breaker_allow(i64)
declare void @circuit_breaker_success(i64)
declare void @circuit_breaker_failure(i64)
declare i64 @circuit_breaker_state(i64)
declare void @circuit_breaker_reset(i64)
declare i64 @retry_policy_new(i64, i64, i64, i64)
declare void @retry_policy_set_jitter(i64, i64)
declare i64 @retry_policy_should_retry(i64)
declare i64 @retry_policy_next_delay(i64)
declare void @retry_policy_reset(i64)
declare i64 @retry_policy_count(i64)
declare i64 @actor_link(i64, i64)
declare void @actor_unlink(i64, i64)
declare i64 @actor_monitor(i64, i64)
declare void @actor_demonitor(i64)
declare void @actor_propagate_exit(i64, i64)
declare i64 @actor_is_linked(i64, i64)
declare i64 @actor_spawn_link(i64, i64, i64)
declare i64 @actor_get_links_count(i64)
declare i64 @actor_send_down(i64, i64, i64)
; Phase 23.1: Supervision Trees
declare i64 @supervisor_new(i64, i64, i64)
declare i64 @supervisor_add_child(i64, i64, i64, i64, i64)
declare i64 @supervisor_start(i64)
declare void @supervisor_stop(i64)
declare i64 @supervisor_handle_exit(i64, i64, i64)
declare i64 @supervisor_child_count(i64)
declare i64 @supervisor_child_status(i64, i64)
declare i64 @supervisor_child_handle(i64, i64)
declare void @supervisor_free(i64)
declare i64 @strategy_one_for_one()
declare i64 @strategy_one_for_all()
declare i64 @strategy_rest_for_one()
declare i64 @child_permanent()
declare i64 @child_temporary()
declare i64 @child_transient()
; Phase 23.2: Work-Stealing Scheduler
declare i64 @scheduler_new(i64)
declare i64 @scheduler_start(i64)
declare i64 @scheduler_submit(i64, i64, i64)
declare i64 @scheduler_submit_local(i64, i64, i64, i64)
declare void @scheduler_stop(i64)
declare void @scheduler_free(i64)
declare i64 @scheduler_worker_count(i64)
declare i64 @scheduler_queue_size(i64)
declare i64 @scheduler_worker_idle(i64, i64)
; Phase 23.3: Lock-Free Mailbox
declare i64 @mailbox_new(i64)
declare i64 @mailbox_send(i64, i64)
declare i64 @mailbox_recv(i64)
declare i64 @mailbox_try_recv(i64)
declare i64 @mailbox_size(i64)
declare i64 @mailbox_empty(i64)
declare i64 @mailbox_full(i64)
declare void @mailbox_close(i64)
declare i64 @mailbox_is_closed(i64)
declare void @mailbox_free(i64)
; Phase 23.6: Actor Discovery and Registry
declare i64 @registry_register(i64, i64)
declare void @registry_unregister(i64)
declare i64 @registry_lookup(i64)
declare i64 @registry_count()
declare i64 @registry_set_metadata(i64, i64)
declare i64 @registry_get_metadata(i64)
; Phase 23.7: Backpressure and Flow Control
declare i64 @flow_controller_new(i64, i64, i64)
declare i64 @flow_check(i64)
declare i64 @flow_acquire(i64)
declare void @flow_release(i64)
declare i64 @flow_is_signaling(i64)
declare i64 @flow_current(i64)
declare i64 @flow_high_watermark(i64)
declare i64 @flow_low_watermark(i64)
declare void @flow_reset(i64)
declare void @flow_free(i64)
declare i64 @flow_mode_drop()
declare i64 @flow_mode_block()
declare i64 @flow_mode_signal()
declare void @intrinsic_exit(i64)
; Phase 1 Stdlib: Option
declare ptr @"option_some"(i64)
declare ptr @"option_none"()
declare i8 @"option_is_some"(i64)
declare i8 @"option_is_none"(i64)
declare i64 @"option_unwrap"(i64)
declare i64 @"option_expect"(i64, i64)
declare i64 @"option_unwrap_or"(i64, i64)
declare i64 @"option_map"(i64, i64)
; Phase 1 Stdlib: Result
declare ptr @"result_ok"(i64)
declare ptr @"result_err"(i64)
declare i8 @"result_is_ok"(i64)
declare i8 @"result_is_err"(i64)
declare i64 @"result_unwrap"(i64)
declare i64 @"result_unwrap_err"(i64)
declare i64 @"result_unwrap_or"(i64, i64)
declare i64 @"result_ok_or"(i64, i64)
; Phase 1 Stdlib: Vec extensions
declare i64 @"vec_sum"(i64)
declare i64 @"vec_find"(i64, i64)
declare i8 @"vec_contains"(i64, i64)
declare i64 @"vec_reverse"(i64)
declare i64 @"vec_clone"(i64)
declare i64 @"vec_first"(i64)
declare i64 @"vec_last"(i64)
declare i64 @"vec_pop"(i64)
declare void @"vec_set"(i64, i64, i64)
declare void @"vec_clear"(i64)
declare void @"vec_remove"(i64, i64)
; Phase 1 Stdlib: HashMap
declare i64 @"hashmap_new"()
declare i64 @"hashmap_insert"(i64, i64, i64)
declare i64 @"hashmap_get"(i64, i64)
declare i64 @"hashmap_remove"(i64, i64)
declare i8 @"hashmap_contains"(i64, i64)
declare i64 @"hashmap_len"(i64)
declare void @"hashmap_clear"(i64)
declare i64 @"hashmap_keys"(i64)
declare i64 @"hashmap_values"(i64)
; Phase 1 Stdlib: HashSet
declare i64 @"hashset_new"()
declare i8 @"hashset_insert"(i64, i64)
declare i8 @"hashset_remove"(i64, i64)
declare i8 @"hashset_contains"(i64, i64)
declare i64 @"hashset_len"(i64)
declare void @"hashset_clear"(i64)
; Phase 1 Stdlib: JSON
declare i64 @"json_parse_simple"(i64)
declare i64 @"json_stringify"(i64)
declare i64 @"json_get_sx"(i64, i64)
declare i64 @"json_keys"(i64)
declare i8 @"json_is_string"(i64)
declare i8 @"json_is_object"(i64)
declare i8 @"json_is_array"(i64)
declare i64 @"json_as_string"(i64)
declare i64 @"json_as_array"(i64)
declare i64 @"json_object_new"()
declare i64 @"json_array_new"()
declare void @"json_object_set"(i64, i64, i64)
declare void @"json_array_push"(i64, i64)
declare i64 @"json_string"(i64)
declare i64 @"json_string_sx"(i64)
declare i64 @"json_array_len"(i64)
declare i64 @"json_object_len"(i64)
declare i64 @"json_as_i64"(i64)
declare i64 @"json_get_index"(i64, i64)
declare i8 @"json_is_null"(i64)
declare i64 @"json_object_key_at"(i64, i64)
declare i64 @"json_object_value_at"(i64, i64)
declare void @"json_object_set_sx"(i64, i64, i64)
; Phase G: AI Features
declare i64 @model_infer(i64)
declare i64 @model_load(i64)
declare i64 @model_embed(i64, i64)
declare i64 @model_classify(i64, i64)
declare void @model_unload(i64)
; AI Vector Operations
declare i64 @vector_new(i64, i64)
declare i64 @vector_len(i64)
declare double @vector_get(i64, i64)
declare void @vector_set(i64, i64, double)
declare double @vector_dot(i64, i64)
declare i64 @vector_add(i64, i64)
declare i64 @vector_sub(i64, i64)
declare i64 @vector_scale(i64, double)
declare double @vector_norm(i64)
declare i64 @vector_normalize(i64)
declare double @vector_cosine_similarity(i64, i64)
; AI Tensor Operations
declare i64 @tensor_new(i64, i64)
declare i64 @tensor_shape(i64)
declare i64 @tensor_rank(i64)
declare double @tensor_get(i64, i64)
declare void @tensor_set(i64, i64, double)
declare i64 @tensor_matmul(i64, i64)
declare i64 @tensor_add(i64, i64)
declare i64 @tensor_transpose(i64)
declare i64 @tensor_reshape(i64, i64)
; Specialist Operations
declare i64 @specialist_query(i64, i64)
declare i64 @specialist_stream(i64, i64)
declare void @specialist_set_context(i64, i64)
declare i64 @specialist_get_context(i64)
; Hive Operations
declare i64 @hive_route(i64, i64)
declare void @hive_add_specialist(i64, i64)
declare void @hive_remove_specialist(i64, i64)
declare i64 @hive_consensus(i64, i64)

define i64 @"VERSION"() {
entry:
  %t0 = call ptr @intrinsic_string_new(ptr @.str.main.0)
  %t1 = ptrtoint ptr %t0 to i64
  ret i64 %t1
}

define i64 @"cli_new"() {
entry:
  %local.cli.2 = alloca i64
  %t3 = call ptr @malloc(i64 72)
  %t4 = ptrtoint ptr %t3 to i64
  store i64 %t4, ptr %local.cli.2
  %t5 = load i64, ptr %local.cli.2
  %t6 = inttoptr i64 %t5 to ptr
  %t7 = call ptr @store_i64(ptr %t6, i64 0, i64 0)
  %t8 = ptrtoint ptr %t7 to i64
  %t9 = load i64, ptr %local.cli.2
  %t10 = call ptr @intrinsic_vec_new()
  %t11 = ptrtoint ptr %t10 to i64
  %t12 = inttoptr i64 %t9 to ptr
  %t13 = inttoptr i64 %t11 to ptr
  %t14 = call ptr @store_ptr(ptr %t12, i64 1, ptr %t13)
  %t15 = ptrtoint ptr %t14 to i64
  %t16 = load i64, ptr %local.cli.2
  %t17 = inttoptr i64 %t16 to ptr
  %t18 = inttoptr i64 0 to ptr
  %t19 = call ptr @store_ptr(ptr %t17, i64 2, ptr %t18)
  %t20 = ptrtoint ptr %t19 to i64
  %t21 = load i64, ptr %local.cli.2
  %t22 = inttoptr i64 %t21 to ptr
  %t23 = call ptr @store_i64(ptr %t22, i64 3, i64 0)
  %t24 = ptrtoint ptr %t23 to i64
  %t25 = load i64, ptr %local.cli.2
  %t26 = inttoptr i64 %t25 to ptr
  %t27 = call ptr @store_i64(ptr %t26, i64 4, i64 0)
  %t28 = ptrtoint ptr %t27 to i64
  %t29 = load i64, ptr %local.cli.2
  %t30 = inttoptr i64 %t29 to ptr
  %t31 = call ptr @store_i64(ptr %t30, i64 5, i64 0)
  %t32 = ptrtoint ptr %t31 to i64
  %t33 = load i64, ptr %local.cli.2
  %t34 = inttoptr i64 %t33 to ptr
  %t35 = call ptr @store_i64(ptr %t34, i64 6, i64 0)
  %t36 = ptrtoint ptr %t35 to i64
  %t37 = load i64, ptr %local.cli.2
  %t38 = inttoptr i64 %t37 to ptr
  %t39 = call ptr @store_i64(ptr %t38, i64 7, i64 0)
  %t40 = ptrtoint ptr %t39 to i64
  %t41 = load i64, ptr %local.cli.2
  %t42 = inttoptr i64 %t41 to ptr
  %t43 = call ptr @store_i64(ptr %t42, i64 8, i64 0)
  %t44 = ptrtoint ptr %t43 to i64
  %t45 = load i64, ptr %local.cli.2
  ret i64 %t45
}

define i64 @"cli_command"(i64 %cli) {
entry:
  %local.cli = alloca i64
  store i64 %cli, ptr %local.cli
  %t46 = load i64, ptr %local.cli
  %t47 = inttoptr i64 %t46 to ptr
  %t48 = call i64 @load_i64(ptr %t47, i64 0)
  ret i64 %t48
}

define i64 @"cli_set_command"(i64 %cli, i64 %cmd) {
entry:
  %local.cli = alloca i64
  store i64 %cli, ptr %local.cli
  %local.cmd = alloca i64
  store i64 %cmd, ptr %local.cmd
  %t49 = load i64, ptr %local.cli
  %t50 = load i64, ptr %local.cmd
  %t51 = inttoptr i64 %t49 to ptr
  %t52 = call ptr @store_i64(ptr %t51, i64 0, i64 %t50)
  %t53 = ptrtoint ptr %t52 to i64
  ret i64 0
}

define i64 @"cli_input_files"(i64 %cli) {
entry:
  %local.cli = alloca i64
  store i64 %cli, ptr %local.cli
  %t54 = load i64, ptr %local.cli
  %t55 = inttoptr i64 %t54 to ptr
  %t56 = call ptr @load_ptr(ptr %t55, i64 1)
  %t57 = ptrtoint ptr %t56 to i64
  ret i64 %t57
}

define i64 @"cli_output_file"(i64 %cli) {
entry:
  %local.cli = alloca i64
  store i64 %cli, ptr %local.cli
  %t58 = load i64, ptr %local.cli
  %t59 = inttoptr i64 %t58 to ptr
  %t60 = call ptr @load_ptr(ptr %t59, i64 2)
  %t61 = ptrtoint ptr %t60 to i64
  ret i64 %t61
}

define i64 @"cli_set_output"(i64 %cli, i64 %path) {
entry:
  %local.cli = alloca i64
  store i64 %cli, ptr %local.cli
  %local.path = alloca i64
  store i64 %path, ptr %local.path
  %t62 = load i64, ptr %local.cli
  %t63 = load i64, ptr %local.path
  %t64 = inttoptr i64 %t62 to ptr
  %t65 = inttoptr i64 %t63 to ptr
  %t66 = call ptr @store_ptr(ptr %t64, i64 2, ptr %t65)
  %t67 = ptrtoint ptr %t66 to i64
  ret i64 0
}

define i64 @"cli_emit_type"(i64 %cli) {
entry:
  %local.cli = alloca i64
  store i64 %cli, ptr %local.cli
  %t68 = load i64, ptr %local.cli
  %t69 = inttoptr i64 %t68 to ptr
  %t70 = call i64 @load_i64(ptr %t69, i64 3)
  ret i64 %t70
}

define i64 @"cli_set_emit"(i64 %cli, i64 %emit) {
entry:
  %local.cli = alloca i64
  store i64 %cli, ptr %local.cli
  %local.emit = alloca i64
  store i64 %emit, ptr %local.emit
  %t71 = load i64, ptr %local.cli
  %t72 = load i64, ptr %local.emit
  %t73 = inttoptr i64 %t71 to ptr
  %t74 = call ptr @store_i64(ptr %t73, i64 3, i64 %t72)
  %t75 = ptrtoint ptr %t74 to i64
  ret i64 0
}

define i64 @"cli_verbose"(i64 %cli) {
entry:
  %local.cli = alloca i64
  store i64 %cli, ptr %local.cli
  %t76 = load i64, ptr %local.cli
  %t77 = inttoptr i64 %t76 to ptr
  %t78 = call i64 @load_i64(ptr %t77, i64 4)
  ret i64 %t78
}

define i64 @"cli_set_verbose"(i64 %cli, i64 %v) {
entry:
  %local.cli = alloca i64
  store i64 %cli, ptr %local.cli
  %local.v = alloca i64
  store i64 %v, ptr %local.v
  %t79 = load i64, ptr %local.cli
  %t80 = load i64, ptr %local.v
  %t81 = inttoptr i64 %t79 to ptr
  %t82 = call ptr @store_i64(ptr %t81, i64 4, i64 %t80)
  %t83 = ptrtoint ptr %t82 to i64
  ret i64 0
}

define i64 @"cli_force"(i64 %cli) {
entry:
  %local.cli = alloca i64
  store i64 %cli, ptr %local.cli
  %t84 = load i64, ptr %local.cli
  %t85 = inttoptr i64 %t84 to ptr
  %t86 = call i64 @load_i64(ptr %t85, i64 5)
  ret i64 %t86
}

define i64 @"cli_set_force"(i64 %cli, i64 %f) {
entry:
  %local.cli = alloca i64
  store i64 %cli, ptr %local.cli
  %local.f = alloca i64
  store i64 %f, ptr %local.f
  %t87 = load i64, ptr %local.cli
  %t88 = load i64, ptr %local.f
  %t89 = inttoptr i64 %t87 to ptr
  %t90 = call ptr @store_i64(ptr %t89, i64 5, i64 %t88)
  %t91 = ptrtoint ptr %t90 to i64
  ret i64 0
}

define i64 @"cli_show_deps"(i64 %cli) {
entry:
  %local.cli = alloca i64
  store i64 %cli, ptr %local.cli
  %t92 = load i64, ptr %local.cli
  %t93 = inttoptr i64 %t92 to ptr
  %t94 = call i64 @load_i64(ptr %t93, i64 6)
  ret i64 %t94
}

define i64 @"cli_set_show_deps"(i64 %cli, i64 %d) {
entry:
  %local.cli = alloca i64
  store i64 %cli, ptr %local.cli
  %local.d = alloca i64
  store i64 %d, ptr %local.d
  %t95 = load i64, ptr %local.cli
  %t96 = load i64, ptr %local.d
  %t97 = inttoptr i64 %t95 to ptr
  %t98 = call ptr @store_i64(ptr %t97, i64 6, i64 %t96)
  %t99 = ptrtoint ptr %t98 to i64
  ret i64 0
}

define i64 @"cli_auto_deps"(i64 %cli) {
entry:
  %local.cli = alloca i64
  store i64 %cli, ptr %local.cli
  %t100 = load i64, ptr %local.cli
  %t101 = inttoptr i64 %t100 to ptr
  %t102 = call i64 @load_i64(ptr %t101, i64 7)
  ret i64 %t102
}

define i64 @"cli_set_auto_deps"(i64 %cli, i64 %a) {
entry:
  %local.cli = alloca i64
  store i64 %cli, ptr %local.cli
  %local.a = alloca i64
  store i64 %a, ptr %local.a
  %t103 = load i64, ptr %local.cli
  %t104 = load i64, ptr %local.a
  %t105 = inttoptr i64 %t103 to ptr
  %t106 = call ptr @store_i64(ptr %t105, i64 7, i64 %t104)
  %t107 = ptrtoint ptr %t106 to i64
  ret i64 0
}

define i64 @"cli_debug_info"(i64 %cli) {
entry:
  %local.cli = alloca i64
  store i64 %cli, ptr %local.cli
  %t108 = load i64, ptr %local.cli
  %t109 = inttoptr i64 %t108 to ptr
  %t110 = call i64 @load_i64(ptr %t109, i64 8)
  ret i64 %t110
}

define i64 @"cli_set_debug_info"(i64 %cli, i64 %g) {
entry:
  %local.cli = alloca i64
  store i64 %cli, ptr %local.cli
  %local.g = alloca i64
  store i64 %g, ptr %local.g
  %t111 = load i64, ptr %local.cli
  %t112 = load i64, ptr %local.g
  %t113 = inttoptr i64 %t111 to ptr
  %t114 = call ptr @store_i64(ptr %t113, i64 8, i64 %t112)
  %t115 = ptrtoint ptr %t114 to i64
  ret i64 0
}

define i64 @"show_help"() {
entry:
  %t116 = call ptr @intrinsic_string_new(ptr @.str.main.1)
  %t117 = ptrtoint ptr %t116 to i64
  %t118 = inttoptr i64 %t117 to ptr
  call void @intrinsic_println(ptr %t118)
  %t119 = call ptr @intrinsic_string_new(ptr @.str.main.2)
  %t120 = ptrtoint ptr %t119 to i64
  %t121 = inttoptr i64 %t120 to ptr
  call void @intrinsic_println(ptr %t121)
  %t122 = call ptr @intrinsic_string_new(ptr @.str.main.3)
  %t123 = ptrtoint ptr %t122 to i64
  %t124 = inttoptr i64 %t123 to ptr
  call void @intrinsic_println(ptr %t124)
  %t125 = call ptr @intrinsic_string_new(ptr @.str.main.4)
  %t126 = ptrtoint ptr %t125 to i64
  %t127 = inttoptr i64 %t126 to ptr
  call void @intrinsic_println(ptr %t127)
  %t128 = call ptr @intrinsic_string_new(ptr @.str.main.5)
  %t129 = ptrtoint ptr %t128 to i64
  %t130 = inttoptr i64 %t129 to ptr
  call void @intrinsic_println(ptr %t130)
  %t131 = call ptr @intrinsic_string_new(ptr @.str.main.6)
  %t132 = ptrtoint ptr %t131 to i64
  %t133 = inttoptr i64 %t132 to ptr
  call void @intrinsic_println(ptr %t133)
  %t134 = call ptr @intrinsic_string_new(ptr @.str.main.7)
  %t135 = ptrtoint ptr %t134 to i64
  %t136 = inttoptr i64 %t135 to ptr
  call void @intrinsic_println(ptr %t136)
  %t137 = call ptr @intrinsic_string_new(ptr @.str.main.8)
  %t138 = ptrtoint ptr %t137 to i64
  %t139 = inttoptr i64 %t138 to ptr
  call void @intrinsic_println(ptr %t139)
  %t140 = call ptr @intrinsic_string_new(ptr @.str.main.9)
  %t141 = ptrtoint ptr %t140 to i64
  %t142 = inttoptr i64 %t141 to ptr
  call void @intrinsic_println(ptr %t142)
  %t143 = call ptr @intrinsic_string_new(ptr @.str.main.10)
  %t144 = ptrtoint ptr %t143 to i64
  %t145 = inttoptr i64 %t144 to ptr
  call void @intrinsic_println(ptr %t145)
  %t146 = call ptr @intrinsic_string_new(ptr @.str.main.11)
  %t147 = ptrtoint ptr %t146 to i64
  %t148 = inttoptr i64 %t147 to ptr
  call void @intrinsic_println(ptr %t148)
  %t149 = call ptr @intrinsic_string_new(ptr @.str.main.12)
  %t150 = ptrtoint ptr %t149 to i64
  %t151 = inttoptr i64 %t150 to ptr
  call void @intrinsic_println(ptr %t151)
  %t152 = call ptr @intrinsic_string_new(ptr @.str.main.13)
  %t153 = ptrtoint ptr %t152 to i64
  %t154 = inttoptr i64 %t153 to ptr
  call void @intrinsic_println(ptr %t154)
  %t155 = call ptr @intrinsic_string_new(ptr @.str.main.14)
  %t156 = ptrtoint ptr %t155 to i64
  %t157 = inttoptr i64 %t156 to ptr
  call void @intrinsic_println(ptr %t157)
  %t158 = call ptr @intrinsic_string_new(ptr @.str.main.15)
  %t159 = ptrtoint ptr %t158 to i64
  %t160 = inttoptr i64 %t159 to ptr
  call void @intrinsic_println(ptr %t160)
  %t161 = call ptr @intrinsic_string_new(ptr @.str.main.16)
  %t162 = ptrtoint ptr %t161 to i64
  %t163 = inttoptr i64 %t162 to ptr
  call void @intrinsic_println(ptr %t163)
  %t164 = call ptr @intrinsic_string_new(ptr @.str.main.17)
  %t165 = ptrtoint ptr %t164 to i64
  %t166 = inttoptr i64 %t165 to ptr
  call void @intrinsic_println(ptr %t166)
  %t167 = call ptr @intrinsic_string_new(ptr @.str.main.18)
  %t168 = ptrtoint ptr %t167 to i64
  %t169 = inttoptr i64 %t168 to ptr
  call void @intrinsic_println(ptr %t169)
  %t170 = call ptr @intrinsic_string_new(ptr @.str.main.19)
  %t171 = ptrtoint ptr %t170 to i64
  %t172 = inttoptr i64 %t171 to ptr
  call void @intrinsic_println(ptr %t172)
  %t173 = call ptr @intrinsic_string_new(ptr @.str.main.20)
  %t174 = ptrtoint ptr %t173 to i64
  %t175 = inttoptr i64 %t174 to ptr
  call void @intrinsic_println(ptr %t175)
  ret i64 0
}

define i64 @"show_version"() {
entry:
  %t176 = call ptr @intrinsic_string_new(ptr @.str.main.21)
  %t177 = ptrtoint ptr %t176 to i64
  %t178 = call i64 @"VERSION"()
  %t179 = inttoptr i64 %t177 to ptr
  %t180 = inttoptr i64 %t178 to ptr
  %t181 = call ptr @intrinsic_string_concat(ptr %t179, ptr %t180)
  %t182 = ptrtoint ptr %t181 to i64
  %t183 = inttoptr i64 %t182 to ptr
  call void @intrinsic_println(ptr %t183)
  ret i64 0
}

define i64 @"CMD_BUILD"() {
entry:
  ret i64 0
}

define i64 @"CMD_RUN"() {
entry:
  ret i64 1
}

define i64 @"CMD_CHECK"() {
entry:
  ret i64 2
}

define i64 @"CMD_HELP"() {
entry:
  ret i64 3
}

define i64 @"CMD_VERSION"() {
entry:
  ret i64 4
}

define i64 @"CMD_REPL"() {
entry:
  ret i64 5
}

define i64 @"CMD_FMT"() {
entry:
  ret i64 6
}

define i64 @"EMIT_LLVM_IR"() {
entry:
  ret i64 0
}

define i64 @"EMIT_OBJ"() {
entry:
  ret i64 1
}

define i64 @"EMIT_EXE"() {
entry:
  ret i64 2
}

define i64 @"EMIT_ASM"() {
entry:
  ret i64 3
}

define i64 @"EMIT_DYLIB"() {
entry:
  ret i64 4
}

define i64 @"parse_emit_type"(i64 %s) {
entry:
  %local.s = alloca i64
  store i64 %s, ptr %local.s
  %t184 = load i64, ptr %local.s
  %t185 = call ptr @intrinsic_string_new(ptr @.str.main.22)
  %t186 = ptrtoint ptr %t185 to i64
  %t187 = inttoptr i64 %t184 to ptr
  %t188 = inttoptr i64 %t186 to ptr
  %t189 = call i1 @intrinsic_string_eq(ptr %t187, ptr %t188)
  %t190 = zext i1 %t189 to i64
  %t191 = icmp ne i64 %t190, 0
  br i1 %t191, label %then0, label %else0
then0:
  %t192 = call i64 @"EMIT_LLVM_IR"()
  ret i64 %t192
  br label %then0_end
then0_end:
  br label %endif0
else0:
  br label %else0_end
else0_end:
  br label %endif0
endif0:
  %t193 = phi i64 [ 0, %then0_end ], [ 0, %else0_end ]
  %t194 = load i64, ptr %local.s
  %t195 = call ptr @intrinsic_string_new(ptr @.str.main.23)
  %t196 = ptrtoint ptr %t195 to i64
  %t197 = inttoptr i64 %t194 to ptr
  %t198 = inttoptr i64 %t196 to ptr
  %t199 = call i1 @intrinsic_string_eq(ptr %t197, ptr %t198)
  %t200 = zext i1 %t199 to i64
  %t201 = icmp ne i64 %t200, 0
  br i1 %t201, label %then1, label %else1
then1:
  %t202 = call i64 @"EMIT_OBJ"()
  ret i64 %t202
  br label %then1_end
then1_end:
  br label %endif1
else1:
  br label %else1_end
else1_end:
  br label %endif1
endif1:
  %t203 = phi i64 [ 0, %then1_end ], [ 0, %else1_end ]
  %t204 = load i64, ptr %local.s
  %t205 = call ptr @intrinsic_string_new(ptr @.str.main.24)
  %t206 = ptrtoint ptr %t205 to i64
  %t207 = inttoptr i64 %t204 to ptr
  %t208 = inttoptr i64 %t206 to ptr
  %t209 = call i1 @intrinsic_string_eq(ptr %t207, ptr %t208)
  %t210 = zext i1 %t209 to i64
  %t211 = icmp ne i64 %t210, 0
  br i1 %t211, label %then2, label %else2
then2:
  %t212 = call i64 @"EMIT_EXE"()
  ret i64 %t212
  br label %then2_end
then2_end:
  br label %endif2
else2:
  br label %else2_end
else2_end:
  br label %endif2
endif2:
  %t213 = phi i64 [ 0, %then2_end ], [ 0, %else2_end ]
  %t214 = load i64, ptr %local.s
  %t215 = call ptr @intrinsic_string_new(ptr @.str.main.25)
  %t216 = ptrtoint ptr %t215 to i64
  %t217 = inttoptr i64 %t214 to ptr
  %t218 = inttoptr i64 %t216 to ptr
  %t219 = call i1 @intrinsic_string_eq(ptr %t217, ptr %t218)
  %t220 = zext i1 %t219 to i64
  %t221 = icmp ne i64 %t220, 0
  br i1 %t221, label %then3, label %else3
then3:
  %t222 = call i64 @"EMIT_ASM"()
  ret i64 %t222
  br label %then3_end
then3_end:
  br label %endif3
else3:
  br label %else3_end
else3_end:
  br label %endif3
endif3:
  %t223 = phi i64 [ 0, %then3_end ], [ 0, %else3_end ]
  %t224 = load i64, ptr %local.s
  %t225 = call ptr @intrinsic_string_new(ptr @.str.main.26)
  %t226 = ptrtoint ptr %t225 to i64
  %t227 = inttoptr i64 %t224 to ptr
  %t228 = inttoptr i64 %t226 to ptr
  %t229 = call i1 @intrinsic_string_eq(ptr %t227, ptr %t228)
  %t230 = zext i1 %t229 to i64
  %t231 = icmp ne i64 %t230, 0
  br i1 %t231, label %then4, label %else4
then4:
  %t232 = call i64 @"EMIT_DYLIB"()
  ret i64 %t232
  br label %then4_end
then4_end:
  br label %endif4
else4:
  br label %else4_end
else4_end:
  br label %endif4
endif4:
  %t233 = phi i64 [ 0, %then4_end ], [ 0, %else4_end ]
  %t234 = sub i64 %t233, 1
  ret i64 %t234
}

define i64 @"parse_command"(i64 %s) {
entry:
  %local.s = alloca i64
  store i64 %s, ptr %local.s
  %t235 = load i64, ptr %local.s
  %t236 = call ptr @intrinsic_string_new(ptr @.str.main.27)
  %t237 = ptrtoint ptr %t236 to i64
  %t238 = inttoptr i64 %t235 to ptr
  %t239 = inttoptr i64 %t237 to ptr
  %t240 = call i1 @intrinsic_string_eq(ptr %t238, ptr %t239)
  %t241 = zext i1 %t240 to i64
  %t242 = icmp ne i64 %t241, 0
  br i1 %t242, label %then5, label %else5
then5:
  %t243 = call i64 @"CMD_BUILD"()
  ret i64 %t243
  br label %then5_end
then5_end:
  br label %endif5
else5:
  br label %else5_end
else5_end:
  br label %endif5
endif5:
  %t244 = phi i64 [ 0, %then5_end ], [ 0, %else5_end ]
  %t245 = load i64, ptr %local.s
  %t246 = call ptr @intrinsic_string_new(ptr @.str.main.28)
  %t247 = ptrtoint ptr %t246 to i64
  %t248 = inttoptr i64 %t245 to ptr
  %t249 = inttoptr i64 %t247 to ptr
  %t250 = call i1 @intrinsic_string_eq(ptr %t248, ptr %t249)
  %t251 = zext i1 %t250 to i64
  %t252 = icmp ne i64 %t251, 0
  br i1 %t252, label %then6, label %else6
then6:
  %t253 = call i64 @"CMD_RUN"()
  ret i64 %t253
  br label %then6_end
then6_end:
  br label %endif6
else6:
  br label %else6_end
else6_end:
  br label %endif6
endif6:
  %t254 = phi i64 [ 0, %then6_end ], [ 0, %else6_end ]
  %t255 = load i64, ptr %local.s
  %t256 = call ptr @intrinsic_string_new(ptr @.str.main.29)
  %t257 = ptrtoint ptr %t256 to i64
  %t258 = inttoptr i64 %t255 to ptr
  %t259 = inttoptr i64 %t257 to ptr
  %t260 = call i1 @intrinsic_string_eq(ptr %t258, ptr %t259)
  %t261 = zext i1 %t260 to i64
  %t262 = icmp ne i64 %t261, 0
  br i1 %t262, label %then7, label %else7
then7:
  %t263 = call i64 @"CMD_CHECK"()
  ret i64 %t263
  br label %then7_end
then7_end:
  br label %endif7
else7:
  br label %else7_end
else7_end:
  br label %endif7
endif7:
  %t264 = phi i64 [ 0, %then7_end ], [ 0, %else7_end ]
  %t265 = load i64, ptr %local.s
  %t266 = call ptr @intrinsic_string_new(ptr @.str.main.30)
  %t267 = ptrtoint ptr %t266 to i64
  %t268 = inttoptr i64 %t265 to ptr
  %t269 = inttoptr i64 %t267 to ptr
  %t270 = call i1 @intrinsic_string_eq(ptr %t268, ptr %t269)
  %t271 = zext i1 %t270 to i64
  %t272 = icmp ne i64 %t271, 0
  br i1 %t272, label %then8, label %else8
then8:
  %t273 = call i64 @"CMD_REPL"()
  ret i64 %t273
  br label %then8_end
then8_end:
  br label %endif8
else8:
  br label %else8_end
else8_end:
  br label %endif8
endif8:
  %t274 = phi i64 [ 0, %then8_end ], [ 0, %else8_end ]
  %t275 = load i64, ptr %local.s
  %t276 = call ptr @intrinsic_string_new(ptr @.str.main.31)
  %t277 = ptrtoint ptr %t276 to i64
  %t278 = inttoptr i64 %t275 to ptr
  %t279 = inttoptr i64 %t277 to ptr
  %t280 = call i1 @intrinsic_string_eq(ptr %t278, ptr %t279)
  %t281 = zext i1 %t280 to i64
  %t282 = icmp ne i64 %t281, 0
  br i1 %t282, label %then9, label %else9
then9:
  %t283 = call i64 @"CMD_FMT"()
  ret i64 %t283
  br label %then9_end
then9_end:
  br label %endif9
else9:
  br label %else9_end
else9_end:
  br label %endif9
endif9:
  %t284 = phi i64 [ 0, %then9_end ], [ 0, %else9_end ]
  %t285 = sub i64 %t284, 1
  ret i64 %t285
}

define i64 @"parse_flag"(i64 %arg) {
entry:
  %local.arg = alloca i64
  store i64 %arg, ptr %local.arg
  %t286 = load i64, ptr %local.arg
  %t287 = call ptr @intrinsic_string_new(ptr @.str.main.32)
  %t288 = ptrtoint ptr %t287 to i64
  %t289 = inttoptr i64 %t286 to ptr
  %t290 = inttoptr i64 %t288 to ptr
  %t291 = call i1 @intrinsic_string_eq(ptr %t289, ptr %t290)
  %t292 = zext i1 %t291 to i64
  %t293 = icmp ne i64 %t292, 0
  br i1 %t293, label %then10, label %else10
then10:
  ret i64 1
  br label %then10_end
then10_end:
  br label %endif10
else10:
  br label %else10_end
else10_end:
  br label %endif10
endif10:
  %t294 = phi i64 [ 0, %then10_end ], [ 0, %else10_end ]
  %t295 = load i64, ptr %local.arg
  %t296 = call ptr @intrinsic_string_new(ptr @.str.main.33)
  %t297 = ptrtoint ptr %t296 to i64
  %t298 = inttoptr i64 %t295 to ptr
  %t299 = inttoptr i64 %t297 to ptr
  %t300 = call i1 @intrinsic_string_eq(ptr %t298, ptr %t299)
  %t301 = zext i1 %t300 to i64
  %t302 = icmp ne i64 %t301, 0
  br i1 %t302, label %then11, label %else11
then11:
  ret i64 1
  br label %then11_end
then11_end:
  br label %endif11
else11:
  br label %else11_end
else11_end:
  br label %endif11
endif11:
  %t303 = phi i64 [ 0, %then11_end ], [ 0, %else11_end ]
  %t304 = load i64, ptr %local.arg
  %t305 = call ptr @intrinsic_string_new(ptr @.str.main.34)
  %t306 = ptrtoint ptr %t305 to i64
  %t307 = inttoptr i64 %t304 to ptr
  %t308 = inttoptr i64 %t306 to ptr
  %t309 = call i1 @intrinsic_string_eq(ptr %t307, ptr %t308)
  %t310 = zext i1 %t309 to i64
  %t311 = icmp ne i64 %t310, 0
  br i1 %t311, label %then12, label %else12
then12:
  ret i64 2
  br label %then12_end
then12_end:
  br label %endif12
else12:
  br label %else12_end
else12_end:
  br label %endif12
endif12:
  %t312 = phi i64 [ 0, %then12_end ], [ 0, %else12_end ]
  %t313 = load i64, ptr %local.arg
  %t314 = call ptr @intrinsic_string_new(ptr @.str.main.35)
  %t315 = ptrtoint ptr %t314 to i64
  %t316 = inttoptr i64 %t313 to ptr
  %t317 = inttoptr i64 %t315 to ptr
  %t318 = call i1 @intrinsic_string_eq(ptr %t316, ptr %t317)
  %t319 = zext i1 %t318 to i64
  %t320 = icmp ne i64 %t319, 0
  br i1 %t320, label %then13, label %else13
then13:
  ret i64 3
  br label %then13_end
then13_end:
  br label %endif13
else13:
  br label %else13_end
else13_end:
  br label %endif13
endif13:
  %t321 = phi i64 [ 0, %then13_end ], [ 0, %else13_end ]
  %t322 = load i64, ptr %local.arg
  %t323 = call ptr @intrinsic_string_new(ptr @.str.main.36)
  %t324 = ptrtoint ptr %t323 to i64
  %t325 = inttoptr i64 %t322 to ptr
  %t326 = inttoptr i64 %t324 to ptr
  %t327 = call i1 @intrinsic_string_eq(ptr %t325, ptr %t326)
  %t328 = zext i1 %t327 to i64
  %t329 = icmp ne i64 %t328, 0
  br i1 %t329, label %then14, label %else14
then14:
  ret i64 3
  br label %then14_end
then14_end:
  br label %endif14
else14:
  br label %else14_end
else14_end:
  br label %endif14
endif14:
  %t330 = phi i64 [ 0, %then14_end ], [ 0, %else14_end ]
  %t331 = load i64, ptr %local.arg
  %t332 = call ptr @intrinsic_string_new(ptr @.str.main.37)
  %t333 = ptrtoint ptr %t332 to i64
  %t334 = inttoptr i64 %t331 to ptr
  %t335 = inttoptr i64 %t333 to ptr
  %t336 = call i1 @intrinsic_string_eq(ptr %t334, ptr %t335)
  %t337 = zext i1 %t336 to i64
  %t338 = icmp ne i64 %t337, 0
  br i1 %t338, label %then15, label %else15
then15:
  ret i64 4
  br label %then15_end
then15_end:
  br label %endif15
else15:
  br label %else15_end
else15_end:
  br label %endif15
endif15:
  %t339 = phi i64 [ 0, %then15_end ], [ 0, %else15_end ]
  %t340 = load i64, ptr %local.arg
  %t341 = call ptr @intrinsic_string_new(ptr @.str.main.38)
  %t342 = ptrtoint ptr %t341 to i64
  %t343 = inttoptr i64 %t340 to ptr
  %t344 = inttoptr i64 %t342 to ptr
  %t345 = call i1 @intrinsic_string_eq(ptr %t343, ptr %t344)
  %t346 = zext i1 %t345 to i64
  %t347 = icmp ne i64 %t346, 0
  br i1 %t347, label %then16, label %else16
then16:
  ret i64 4
  br label %then16_end
then16_end:
  br label %endif16
else16:
  br label %else16_end
else16_end:
  br label %endif16
endif16:
  %t348 = phi i64 [ 0, %then16_end ], [ 0, %else16_end ]
  %t349 = load i64, ptr %local.arg
  %t350 = call ptr @intrinsic_string_new(ptr @.str.main.39)
  %t351 = ptrtoint ptr %t350 to i64
  %t352 = inttoptr i64 %t349 to ptr
  %t353 = inttoptr i64 %t351 to ptr
  %t354 = call i1 @intrinsic_string_eq(ptr %t352, ptr %t353)
  %t355 = zext i1 %t354 to i64
  %t356 = icmp ne i64 %t355, 0
  br i1 %t356, label %then17, label %else17
then17:
  ret i64 5
  br label %then17_end
then17_end:
  br label %endif17
else17:
  br label %else17_end
else17_end:
  br label %endif17
endif17:
  %t357 = phi i64 [ 0, %then17_end ], [ 0, %else17_end ]
  %t358 = load i64, ptr %local.arg
  %t359 = call ptr @intrinsic_string_new(ptr @.str.main.40)
  %t360 = ptrtoint ptr %t359 to i64
  %t361 = inttoptr i64 %t358 to ptr
  %t362 = inttoptr i64 %t360 to ptr
  %t363 = call i1 @intrinsic_string_eq(ptr %t361, ptr %t362)
  %t364 = zext i1 %t363 to i64
  %t365 = icmp ne i64 %t364, 0
  br i1 %t365, label %then18, label %else18
then18:
  ret i64 6
  br label %then18_end
then18_end:
  br label %endif18
else18:
  br label %else18_end
else18_end:
  br label %endif18
endif18:
  %t366 = phi i64 [ 0, %then18_end ], [ 0, %else18_end ]
  %t367 = load i64, ptr %local.arg
  %t368 = call ptr @intrinsic_string_new(ptr @.str.main.41)
  %t369 = ptrtoint ptr %t368 to i64
  %t370 = inttoptr i64 %t367 to ptr
  %t371 = inttoptr i64 %t369 to ptr
  %t372 = call i1 @intrinsic_string_eq(ptr %t370, ptr %t371)
  %t373 = zext i1 %t372 to i64
  %t374 = icmp ne i64 %t373, 0
  br i1 %t374, label %then19, label %else19
then19:
  ret i64 7
  br label %then19_end
then19_end:
  br label %endif19
else19:
  br label %else19_end
else19_end:
  br label %endif19
endif19:
  %t375 = phi i64 [ 0, %then19_end ], [ 0, %else19_end ]
  %t376 = sub i64 %t375, 1
  ret i64 %t376
}

define i64 @"arg_takes_value"(i64 %arg) {
entry:
  %local.arg = alloca i64
  store i64 %arg, ptr %local.arg
  %t377 = load i64, ptr %local.arg
  %t378 = call ptr @intrinsic_string_new(ptr @.str.main.42)
  %t379 = ptrtoint ptr %t378 to i64
  %t380 = inttoptr i64 %t377 to ptr
  %t381 = inttoptr i64 %t379 to ptr
  %t382 = call i1 @intrinsic_string_eq(ptr %t380, ptr %t381)
  %t383 = zext i1 %t382 to i64
  %t384 = icmp ne i64 %t383, 0
  br i1 %t384, label %then20, label %else20
then20:
  ret i64 1
  br label %then20_end
then20_end:
  br label %endif20
else20:
  br label %else20_end
else20_end:
  br label %endif20
endif20:
  %t385 = phi i64 [ 0, %then20_end ], [ 0, %else20_end ]
  %t386 = load i64, ptr %local.arg
  %t387 = call ptr @intrinsic_string_new(ptr @.str.main.43)
  %t388 = ptrtoint ptr %t387 to i64
  %t389 = inttoptr i64 %t386 to ptr
  %t390 = inttoptr i64 %t388 to ptr
  %t391 = call i1 @intrinsic_string_eq(ptr %t389, ptr %t390)
  %t392 = zext i1 %t391 to i64
  %t393 = icmp ne i64 %t392, 0
  br i1 %t393, label %then21, label %else21
then21:
  ret i64 2
  br label %then21_end
then21_end:
  br label %endif21
else21:
  br label %else21_end
else21_end:
  br label %endif21
endif21:
  %t394 = phi i64 [ 0, %then21_end ], [ 0, %else21_end ]
  ret i64 0
}

define i64 @"parse_args"(i64 %args) {
entry:
  %local.cli.395 = alloca i64
  %local.argc.396 = alloca i64
  %local.i.397 = alloca i64
  %local.arg.398 = alloca i64
  %local.flag.399 = alloca i64
  %local.val_type.400 = alloca i64
  %local.emit.401 = alloca i64
  %local.cmd.402 = alloca i64
  %local.args = alloca i64
  store i64 %args, ptr %local.args
  %t403 = call i64 @"cli_new"()
  store i64 %t403, ptr %local.cli.395
  %t404 = load i64, ptr %local.args
  %t405 = inttoptr i64 %t404 to ptr
  %t406 = call i64 @intrinsic_vec_len(ptr %t405)
  store i64 %t406, ptr %local.argc.396
  store i64 1, ptr %local.i.397
  br label %loop22
loop22:
  %t407 = load i64, ptr %local.i.397
  %t408 = load i64, ptr %local.argc.396
  %t409 = icmp slt i64 %t407, %t408
  %t410 = zext i1 %t409 to i64
  %t411 = icmp ne i64 %t410, 0
  br i1 %t411, label %body22, label %endloop22
body22:
  %t412 = load i64, ptr %local.args
  %t413 = load i64, ptr %local.i.397
  %t414 = inttoptr i64 %t412 to ptr
  %t415 = call ptr @intrinsic_vec_get(ptr %t414, i64 %t413)
  %t416 = ptrtoint ptr %t415 to i64
  store i64 %t416, ptr %local.arg.398
  %t417 = load i64, ptr %local.arg.398
  %t418 = call i64 @"parse_flag"(i64 %t417)
  store i64 %t418, ptr %local.flag.399
  %t419 = load i64, ptr %local.flag.399
  %t420 = icmp eq i64 %t419, 1
  %t421 = zext i1 %t420 to i64
  %t422 = icmp ne i64 %t421, 0
  br i1 %t422, label %then23, label %else23
then23:
  %t423 = load i64, ptr %local.cli.395
  %t424 = call i64 @"CMD_HELP"()
  %t425 = call i64 @"cli_set_command"(i64 %t423, i64 %t424)
  %t426 = load i64, ptr %local.cli.395
  ret i64 %t426
  br label %then23_end
then23_end:
  br label %endif23
else23:
  br label %else23_end
else23_end:
  br label %endif23
endif23:
  %t427 = phi i64 [ 0, %then23_end ], [ 0, %else23_end ]
  %t428 = load i64, ptr %local.flag.399
  %t429 = icmp eq i64 %t428, 2
  %t430 = zext i1 %t429 to i64
  %t431 = icmp ne i64 %t430, 0
  br i1 %t431, label %then24, label %else24
then24:
  %t432 = load i64, ptr %local.cli.395
  %t433 = call i64 @"CMD_VERSION"()
  %t434 = call i64 @"cli_set_command"(i64 %t432, i64 %t433)
  %t435 = load i64, ptr %local.cli.395
  ret i64 %t435
  br label %then24_end
then24_end:
  br label %endif24
else24:
  br label %else24_end
else24_end:
  br label %endif24
endif24:
  %t436 = phi i64 [ 0, %then24_end ], [ 0, %else24_end ]
  %t437 = load i64, ptr %local.flag.399
  %t438 = icmp eq i64 %t437, 3
  %t439 = zext i1 %t438 to i64
  %t440 = icmp ne i64 %t439, 0
  br i1 %t440, label %then25, label %else25
then25:
  %t441 = load i64, ptr %local.cli.395
  %t442 = call i64 @"cli_set_verbose"(i64 %t441, i64 1)
  %t443 = load i64, ptr %local.i.397
  %t444 = add i64 %t443, 1
  store i64 %t444, ptr %local.i.397
  br label %loop22
after_continue26:
  br label %then25_end
then25_end:
  br label %endif25
else25:
  br label %else25_end
else25_end:
  br label %endif25
endif25:
  %t445 = phi i64 [ 0, %then25_end ], [ 0, %else25_end ]
  %t446 = load i64, ptr %local.flag.399
  %t447 = icmp eq i64 %t446, 4
  %t448 = zext i1 %t447 to i64
  %t449 = icmp ne i64 %t448, 0
  br i1 %t449, label %then27, label %else27
then27:
  %t450 = load i64, ptr %local.cli.395
  %t451 = call i64 @"cli_set_force"(i64 %t450, i64 1)
  %t452 = load i64, ptr %local.i.397
  %t453 = add i64 %t452, 1
  store i64 %t453, ptr %local.i.397
  br label %loop22
after_continue28:
  br label %then27_end
then27_end:
  br label %endif27
else27:
  br label %else27_end
else27_end:
  br label %endif27
endif27:
  %t454 = phi i64 [ 0, %then27_end ], [ 0, %else27_end ]
  %t455 = load i64, ptr %local.flag.399
  %t456 = icmp eq i64 %t455, 5
  %t457 = zext i1 %t456 to i64
  %t458 = icmp ne i64 %t457, 0
  br i1 %t458, label %then29, label %else29
then29:
  %t459 = load i64, ptr %local.cli.395
  %t460 = call i64 @"cli_set_show_deps"(i64 %t459, i64 1)
  %t461 = load i64, ptr %local.i.397
  %t462 = add i64 %t461, 1
  store i64 %t462, ptr %local.i.397
  br label %loop22
after_continue30:
  br label %then29_end
then29_end:
  br label %endif29
else29:
  br label %else29_end
else29_end:
  br label %endif29
endif29:
  %t463 = phi i64 [ 0, %then29_end ], [ 0, %else29_end ]
  %t464 = load i64, ptr %local.flag.399
  %t465 = icmp eq i64 %t464, 6
  %t466 = zext i1 %t465 to i64
  %t467 = icmp ne i64 %t466, 0
  br i1 %t467, label %then31, label %else31
then31:
  %t468 = load i64, ptr %local.cli.395
  %t469 = call i64 @"cli_set_auto_deps"(i64 %t468, i64 1)
  %t470 = load i64, ptr %local.i.397
  %t471 = add i64 %t470, 1
  store i64 %t471, ptr %local.i.397
  br label %loop22
after_continue32:
  br label %then31_end
then31_end:
  br label %endif31
else31:
  br label %else31_end
else31_end:
  br label %endif31
endif31:
  %t472 = phi i64 [ 0, %then31_end ], [ 0, %else31_end ]
  %t473 = load i64, ptr %local.flag.399
  %t474 = icmp eq i64 %t473, 7
  %t475 = zext i1 %t474 to i64
  %t476 = icmp ne i64 %t475, 0
  br i1 %t476, label %then33, label %else33
then33:
  %t477 = load i64, ptr %local.cli.395
  %t478 = call i64 @"cli_set_debug_info"(i64 %t477, i64 1)
  %t479 = load i64, ptr %local.i.397
  %t480 = add i64 %t479, 1
  store i64 %t480, ptr %local.i.397
  br label %loop22
after_continue34:
  br label %then33_end
then33_end:
  br label %endif33
else33:
  br label %else33_end
else33_end:
  br label %endif33
endif33:
  %t481 = phi i64 [ 0, %then33_end ], [ 0, %else33_end ]
  %t482 = load i64, ptr %local.arg.398
  %t483 = call i64 @"arg_takes_value"(i64 %t482)
  store i64 %t483, ptr %local.val_type.400
  %t484 = load i64, ptr %local.val_type.400
  %t485 = icmp eq i64 %t484, 1
  %t486 = zext i1 %t485 to i64
  %t487 = icmp ne i64 %t486, 0
  br i1 %t487, label %then35, label %else35
then35:
  %t488 = load i64, ptr %local.i.397
  %t489 = add i64 %t488, 1
  %t490 = load i64, ptr %local.argc.396
  %t491 = icmp slt i64 %t489, %t490
  %t492 = zext i1 %t491 to i64
  %t493 = icmp ne i64 %t492, 0
  br i1 %t493, label %then36, label %else36
then36:
  %t494 = load i64, ptr %local.i.397
  %t495 = add i64 %t494, 1
  store i64 %t495, ptr %local.i.397
  %t496 = load i64, ptr %local.cli.395
  %t497 = load i64, ptr %local.args
  %t498 = load i64, ptr %local.i.397
  %t499 = inttoptr i64 %t497 to ptr
  %t500 = call ptr @intrinsic_vec_get(ptr %t499, i64 %t498)
  %t501 = ptrtoint ptr %t500 to i64
  %t502 = call i64 @"cli_set_output"(i64 %t496, i64 %t501)
  br label %then36_end
then36_end:
  br label %endif36
else36:
  br label %else36_end
else36_end:
  br label %endif36
endif36:
  %t503 = phi i64 [ 0, %then36_end ], [ 0, %else36_end ]
  %t504 = load i64, ptr %local.i.397
  %t505 = add i64 %t504, 1
  store i64 %t505, ptr %local.i.397
  br label %loop22
after_continue37:
  br label %then35_end
then35_end:
  br label %endif35
else35:
  br label %else35_end
else35_end:
  br label %endif35
endif35:
  %t506 = phi i64 [ 0, %then35_end ], [ 0, %else35_end ]
  %t507 = load i64, ptr %local.val_type.400
  %t508 = icmp eq i64 %t507, 2
  %t509 = zext i1 %t508 to i64
  %t510 = icmp ne i64 %t509, 0
  br i1 %t510, label %then38, label %else38
then38:
  %t511 = load i64, ptr %local.i.397
  %t512 = add i64 %t511, 1
  %t513 = load i64, ptr %local.argc.396
  %t514 = icmp slt i64 %t512, %t513
  %t515 = zext i1 %t514 to i64
  %t516 = icmp ne i64 %t515, 0
  br i1 %t516, label %then39, label %else39
then39:
  %t517 = load i64, ptr %local.i.397
  %t518 = add i64 %t517, 1
  store i64 %t518, ptr %local.i.397
  %t519 = load i64, ptr %local.args
  %t520 = load i64, ptr %local.i.397
  %t521 = inttoptr i64 %t519 to ptr
  %t522 = call ptr @intrinsic_vec_get(ptr %t521, i64 %t520)
  %t523 = ptrtoint ptr %t522 to i64
  %t524 = call i64 @"parse_emit_type"(i64 %t523)
  store i64 %t524, ptr %local.emit.401
  %t525 = load i64, ptr %local.emit.401
  %t526 = icmp sge i64 %t525, 0
  %t527 = zext i1 %t526 to i64
  %t528 = icmp ne i64 %t527, 0
  br i1 %t528, label %then40, label %else40
then40:
  %t529 = load i64, ptr %local.cli.395
  %t530 = load i64, ptr %local.emit.401
  %t531 = call i64 @"cli_set_emit"(i64 %t529, i64 %t530)
  br label %then40_end
then40_end:
  br label %endif40
else40:
  %t532 = call ptr @intrinsic_string_new(ptr @.str.main.44)
  %t533 = ptrtoint ptr %t532 to i64
  %t534 = load i64, ptr %local.args
  %t535 = load i64, ptr %local.i.397
  %t536 = inttoptr i64 %t534 to ptr
  %t537 = call ptr @intrinsic_vec_get(ptr %t536, i64 %t535)
  %t538 = ptrtoint ptr %t537 to i64
  %t539 = inttoptr i64 %t533 to ptr
  %t540 = inttoptr i64 %t538 to ptr
  %t541 = call ptr @intrinsic_string_concat(ptr %t539, ptr %t540)
  %t542 = ptrtoint ptr %t541 to i64
  %t543 = inttoptr i64 %t542 to ptr
  call void @intrinsic_println(ptr %t543)
  br label %else40_end
else40_end:
  br label %endif40
endif40:
  %t544 = phi i64 [ 0, %then40_end ], [ 0, %else40_end ]
  br label %then39_end
then39_end:
  br label %endif39
else39:
  br label %else39_end
else39_end:
  br label %endif39
endif39:
  %t545 = phi i64 [ %t544, %then39_end ], [ 0, %else39_end ]
  %t546 = load i64, ptr %local.i.397
  %t547 = add i64 %t546, 1
  store i64 %t547, ptr %local.i.397
  br label %loop22
after_continue41:
  br label %then38_end
then38_end:
  br label %endif38
else38:
  br label %else38_end
else38_end:
  br label %endif38
endif38:
  %t548 = phi i64 [ 0, %then38_end ], [ 0, %else38_end ]
  %t549 = load i64, ptr %local.arg.398
  %t550 = call i64 @"parse_command"(i64 %t549)
  store i64 %t550, ptr %local.cmd.402
  %t551 = load i64, ptr %local.cmd.402
  %t552 = icmp sge i64 %t551, 0
  %t553 = zext i1 %t552 to i64
  %t554 = icmp ne i64 %t553, 0
  br i1 %t554, label %then42, label %else42
then42:
  %t555 = load i64, ptr %local.cli.395
  %t556 = load i64, ptr %local.cmd.402
  %t557 = call i64 @"cli_set_command"(i64 %t555, i64 %t556)
  %t558 = load i64, ptr %local.i.397
  %t559 = add i64 %t558, 1
  store i64 %t559, ptr %local.i.397
  br label %loop22
after_continue43:
  br label %then42_end
then42_end:
  br label %endif42
else42:
  br label %else42_end
else42_end:
  br label %endif42
endif42:
  %t560 = phi i64 [ 0, %then42_end ], [ 0, %else42_end ]
  %t561 = load i64, ptr %local.cli.395
  %t562 = call i64 @"cli_input_files"(i64 %t561)
  %t563 = load i64, ptr %local.arg.398
  %t564 = inttoptr i64 %t562 to ptr
  %t565 = inttoptr i64 %t563 to ptr
  call void @intrinsic_vec_push(ptr %t564, ptr %t565)
  %t566 = load i64, ptr %local.i.397
  %t567 = add i64 %t566, 1
  store i64 %t567, ptr %local.i.397
  br label %loop22
endloop22:
  %t568 = load i64, ptr %local.cli.395
  ret i64 %t568
}

define i64 @"is_up_to_date"(i64 %source_path, i64 %output_path) {
entry:
  %local.src_mtime.569 = alloca i64
  %local.out_mtime.570 = alloca i64
  %local.source_path = alloca i64
  store i64 %source_path, ptr %local.source_path
  %local.output_path = alloca i64
  store i64 %output_path, ptr %local.output_path
  %t571 = load i64, ptr %local.output_path
  %t572 = inttoptr i64 %t571 to ptr
  %t573 = call i64 @intrinsic_file_exists(ptr %t572)
  %t574 = icmp eq i64 %t573, 0
  %t575 = zext i1 %t574 to i64
  %t576 = icmp ne i64 %t575, 0
  br i1 %t576, label %then44, label %else44
then44:
  ret i64 0
  br label %then44_end
then44_end:
  br label %endif44
else44:
  br label %else44_end
else44_end:
  br label %endif44
endif44:
  %t577 = phi i64 [ 0, %then44_end ], [ 0, %else44_end ]
  %t578 = load i64, ptr %local.source_path
  %t579 = inttoptr i64 %t578 to ptr
  %t580 = call i64 @intrinsic_file_mtime(ptr %t579)
  store i64 %t580, ptr %local.src_mtime.569
  %t581 = load i64, ptr %local.output_path
  %t582 = inttoptr i64 %t581 to ptr
  %t583 = call i64 @intrinsic_file_mtime(ptr %t582)
  store i64 %t583, ptr %local.out_mtime.570
  %t584 = load i64, ptr %local.src_mtime.569
  %t585 = load i64, ptr %local.out_mtime.570
  %t586 = icmp sgt i64 %t584, %t585
  %t587 = zext i1 %t586 to i64
  %t588 = icmp ne i64 %t587, 0
  br i1 %t588, label %then45, label %else45
then45:
  ret i64 0
  br label %then45_end
then45_end:
  br label %endif45
else45:
  br label %else45_end
else45_end:
  br label %endif45
endif45:
  %t589 = phi i64 [ 0, %then45_end ], [ 0, %else45_end ]
  ret i64 1
}

define i64 @"vec_contains_string"(i64 %v, i64 %s) {
entry:
  %local.n.590 = alloca i64
  %local.i.591 = alloca i64
  %local.v = alloca i64
  store i64 %v, ptr %local.v
  %local.s = alloca i64
  store i64 %s, ptr %local.s
  %t592 = load i64, ptr %local.v
  %t593 = inttoptr i64 %t592 to ptr
  %t594 = call i64 @intrinsic_vec_len(ptr %t593)
  store i64 %t594, ptr %local.n.590
  store i64 0, ptr %local.i.591
  br label %loop46
loop46:
  %t595 = load i64, ptr %local.i.591
  %t596 = load i64, ptr %local.n.590
  %t597 = icmp slt i64 %t595, %t596
  %t598 = zext i1 %t597 to i64
  %t599 = icmp ne i64 %t598, 0
  br i1 %t599, label %body46, label %endloop46
body46:
  %t600 = load i64, ptr %local.v
  %t601 = load i64, ptr %local.i.591
  %t602 = inttoptr i64 %t600 to ptr
  %t603 = call ptr @intrinsic_vec_get(ptr %t602, i64 %t601)
  %t604 = ptrtoint ptr %t603 to i64
  %t605 = load i64, ptr %local.s
  %t606 = inttoptr i64 %t604 to ptr
  %t607 = inttoptr i64 %t605 to ptr
  %t608 = call i1 @intrinsic_string_eq(ptr %t606, ptr %t607)
  %t609 = zext i1 %t608 to i64
  %t610 = icmp ne i64 %t609, 0
  br i1 %t610, label %then47, label %else47
then47:
  ret i64 1
  br label %then47_end
then47_end:
  br label %endif47
else47:
  br label %else47_end
else47_end:
  br label %endif47
endif47:
  %t611 = phi i64 [ 0, %then47_end ], [ 0, %else47_end ]
  %t612 = load i64, ptr %local.i.591
  %t613 = add i64 %t612, 1
  store i64 %t613, ptr %local.i.591
  br label %loop46
endloop46:
  ret i64 0
}

define i64 @"extract_dependencies"(i64 %source_path) {
entry:
  %local.deps.614 = alloca i64
  %local.source.615 = alloca i64
  %local.dir.616 = alloca i64
  %local.tokens.617 = alloca i64
  %local.n.618 = alloca i64
  %local.i.619 = alloca i64
  %local.tok.620 = alloca i64
  %local.kind.621 = alloca i64
  %local.text.622 = alloca i64
  %local.name_tok.623 = alloca i64
  %local.mod_name.624 = alloca i64
  %local.dep_path.625 = alloca i64
  %local.path_tok.626 = alloca i64
  %local.path_name.627 = alloca i64
  %local.dep_path.628 = alloca i64
  %local.source_path = alloca i64
  store i64 %source_path, ptr %local.source_path
  %t629 = call ptr @intrinsic_vec_new()
  %t630 = ptrtoint ptr %t629 to i64
  store i64 %t630, ptr %local.deps.614
  %t631 = load i64, ptr %local.source_path
  %t632 = inttoptr i64 %t631 to ptr
  %t633 = call ptr @intrinsic_read_file(ptr %t632)
  %t634 = ptrtoint ptr %t633 to i64
  store i64 %t634, ptr %local.source.615
  %t635 = load i64, ptr %local.source.615
  %t636 = icmp eq i64 %t635, 0
  %t637 = zext i1 %t636 to i64
  %t638 = icmp ne i64 %t637, 0
  br i1 %t638, label %then48, label %else48
then48:
  %t639 = load i64, ptr %local.deps.614
  ret i64 %t639
  br label %then48_end
then48_end:
  br label %endif48
else48:
  br label %else48_end
else48_end:
  br label %endif48
endif48:
  %t640 = phi i64 [ 0, %then48_end ], [ 0, %else48_end ]
  %t641 = load i64, ptr %local.source_path
  %t642 = call i64 @"get_directory"(i64 %t641)
  store i64 %t642, ptr %local.dir.616
  %t643 = load i64, ptr %local.source.615
  %t644 = call i64 @"tokenize"(i64 %t643)
  store i64 %t644, ptr %local.tokens.617
  %t645 = load i64, ptr %local.tokens.617
  %t646 = inttoptr i64 %t645 to ptr
  %t647 = call i64 @intrinsic_vec_len(ptr %t646)
  store i64 %t647, ptr %local.n.618
  store i64 0, ptr %local.i.619
  br label %loop49
loop49:
  %t648 = load i64, ptr %local.i.619
  %t649 = load i64, ptr %local.n.618
  %t650 = icmp slt i64 %t648, %t649
  %t651 = zext i1 %t650 to i64
  %t652 = icmp ne i64 %t651, 0
  br i1 %t652, label %body49, label %endloop49
body49:
  %t653 = load i64, ptr %local.tokens.617
  %t654 = load i64, ptr %local.i.619
  %t655 = inttoptr i64 %t653 to ptr
  %t656 = call ptr @intrinsic_vec_get(ptr %t655, i64 %t654)
  %t657 = ptrtoint ptr %t656 to i64
  store i64 %t657, ptr %local.tok.620
  %t658 = load i64, ptr %local.tok.620
  %t659 = call i64 @"token_kind"(i64 %t658)
  store i64 %t659, ptr %local.kind.621
  %t660 = load i64, ptr %local.tok.620
  %t661 = call i64 @"token_text"(i64 %t660)
  store i64 %t661, ptr %local.text.622
  %t662 = load i64, ptr %local.text.622
  %t663 = call ptr @intrinsic_string_new(ptr @.str.main.45)
  %t664 = ptrtoint ptr %t663 to i64
  %t665 = inttoptr i64 %t662 to ptr
  %t666 = inttoptr i64 %t664 to ptr
  %t667 = call i1 @intrinsic_string_eq(ptr %t665, ptr %t666)
  %t668 = zext i1 %t667 to i64
  %t669 = icmp ne i64 %t668, 0
  br i1 %t669, label %then50, label %else50
then50:
  %t670 = load i64, ptr %local.i.619
  %t671 = add i64 %t670, 1
  %t672 = load i64, ptr %local.n.618
  %t673 = icmp slt i64 %t671, %t672
  %t674 = zext i1 %t673 to i64
  %t675 = icmp ne i64 %t674, 0
  br i1 %t675, label %then51, label %else51
then51:
  %t676 = load i64, ptr %local.tokens.617
  %t677 = load i64, ptr %local.i.619
  %t678 = add i64 %t677, 1
  %t679 = inttoptr i64 %t676 to ptr
  %t680 = call ptr @intrinsic_vec_get(ptr %t679, i64 %t678)
  %t681 = ptrtoint ptr %t680 to i64
  store i64 %t681, ptr %local.name_tok.623
  %t682 = load i64, ptr %local.name_tok.623
  %t683 = call i64 @"token_text"(i64 %t682)
  store i64 %t683, ptr %local.mod_name.624
  %t684 = load i64, ptr %local.dir.616
  %t685 = load i64, ptr %local.mod_name.624
  %t686 = call ptr @intrinsic_string_new(ptr @.str.main.46)
  %t687 = ptrtoint ptr %t686 to i64
  %t688 = inttoptr i64 %t685 to ptr
  %t689 = inttoptr i64 %t687 to ptr
  %t690 = call ptr @intrinsic_string_concat(ptr %t688, ptr %t689)
  %t691 = ptrtoint ptr %t690 to i64
  %t692 = inttoptr i64 %t684 to ptr
  %t693 = inttoptr i64 %t691 to ptr
  %t694 = call ptr @intrinsic_string_concat(ptr %t692, ptr %t693)
  %t695 = ptrtoint ptr %t694 to i64
  store i64 %t695, ptr %local.dep_path.625
  %t696 = load i64, ptr %local.dep_path.625
  %t697 = inttoptr i64 %t696 to ptr
  %t698 = call i64 @intrinsic_file_exists(ptr %t697)
  %t699 = icmp eq i64 %t698, 1
  %t700 = zext i1 %t699 to i64
  %t701 = icmp ne i64 %t700, 0
  br i1 %t701, label %then52, label %else52
then52:
  %t702 = load i64, ptr %local.deps.614
  %t703 = load i64, ptr %local.dep_path.625
  %t704 = call i64 @"vec_contains_string"(i64 %t702, i64 %t703)
  %t705 = icmp eq i64 %t704, 0
  %t706 = zext i1 %t705 to i64
  %t707 = icmp ne i64 %t706, 0
  br i1 %t707, label %then53, label %else53
then53:
  %t708 = load i64, ptr %local.deps.614
  %t709 = load i64, ptr %local.dep_path.625
  %t710 = inttoptr i64 %t708 to ptr
  %t711 = inttoptr i64 %t709 to ptr
  call void @intrinsic_vec_push(ptr %t710, ptr %t711)
  br label %then53_end
then53_end:
  br label %endif53
else53:
  br label %else53_end
else53_end:
  br label %endif53
endif53:
  %t712 = phi i64 [ 0, %then53_end ], [ 0, %else53_end ]
  br label %then52_end
then52_end:
  br label %endif52
else52:
  br label %else52_end
else52_end:
  br label %endif52
endif52:
  %t713 = phi i64 [ %t712, %then52_end ], [ 0, %else52_end ]
  %t714 = load i64, ptr %local.i.619
  %t715 = add i64 %t714, 2
  store i64 %t715, ptr %local.i.619
  br label %then51_end
then51_end:
  br label %endif51
else51:
  %t716 = load i64, ptr %local.i.619
  %t717 = add i64 %t716, 1
  store i64 %t717, ptr %local.i.619
  br label %else51_end
else51_end:
  br label %endif51
endif51:
  %t718 = phi i64 [ 0, %then51_end ], [ 0, %else51_end ]
  br label %then50_end
then50_end:
  br label %endif50
else50:
  %t719 = load i64, ptr %local.text.622
  %t720 = call ptr @intrinsic_string_new(ptr @.str.main.47)
  %t721 = ptrtoint ptr %t720 to i64
  %t722 = inttoptr i64 %t719 to ptr
  %t723 = inttoptr i64 %t721 to ptr
  %t724 = call i1 @intrinsic_string_eq(ptr %t722, ptr %t723)
  %t725 = zext i1 %t724 to i64
  %t726 = icmp ne i64 %t725, 0
  br i1 %t726, label %then54, label %else54
then54:
  %t727 = load i64, ptr %local.i.619
  %t728 = add i64 %t727, 1
  %t729 = load i64, ptr %local.n.618
  %t730 = icmp slt i64 %t728, %t729
  %t731 = zext i1 %t730 to i64
  %t732 = icmp ne i64 %t731, 0
  br i1 %t732, label %then55, label %else55
then55:
  %t733 = load i64, ptr %local.tokens.617
  %t734 = load i64, ptr %local.i.619
  %t735 = add i64 %t734, 1
  %t736 = inttoptr i64 %t733 to ptr
  %t737 = call ptr @intrinsic_vec_get(ptr %t736, i64 %t735)
  %t738 = ptrtoint ptr %t737 to i64
  store i64 %t738, ptr %local.path_tok.626
  %t739 = load i64, ptr %local.path_tok.626
  %t740 = call i64 @"token_text"(i64 %t739)
  store i64 %t740, ptr %local.path_name.627
  %t741 = load i64, ptr %local.dir.616
  %t742 = load i64, ptr %local.path_name.627
  %t743 = call ptr @intrinsic_string_new(ptr @.str.main.48)
  %t744 = ptrtoint ptr %t743 to i64
  %t745 = inttoptr i64 %t742 to ptr
  %t746 = inttoptr i64 %t744 to ptr
  %t747 = call ptr @intrinsic_string_concat(ptr %t745, ptr %t746)
  %t748 = ptrtoint ptr %t747 to i64
  %t749 = inttoptr i64 %t741 to ptr
  %t750 = inttoptr i64 %t748 to ptr
  %t751 = call ptr @intrinsic_string_concat(ptr %t749, ptr %t750)
  %t752 = ptrtoint ptr %t751 to i64
  store i64 %t752, ptr %local.dep_path.628
  %t753 = load i64, ptr %local.dep_path.628
  %t754 = inttoptr i64 %t753 to ptr
  %t755 = call i64 @intrinsic_file_exists(ptr %t754)
  %t756 = icmp eq i64 %t755, 1
  %t757 = zext i1 %t756 to i64
  %t758 = icmp ne i64 %t757, 0
  br i1 %t758, label %then56, label %else56
then56:
  %t759 = load i64, ptr %local.deps.614
  %t760 = load i64, ptr %local.dep_path.628
  %t761 = call i64 @"vec_contains_string"(i64 %t759, i64 %t760)
  %t762 = icmp eq i64 %t761, 0
  %t763 = zext i1 %t762 to i64
  %t764 = icmp ne i64 %t763, 0
  br i1 %t764, label %then57, label %else57
then57:
  %t765 = load i64, ptr %local.deps.614
  %t766 = load i64, ptr %local.dep_path.628
  %t767 = inttoptr i64 %t765 to ptr
  %t768 = inttoptr i64 %t766 to ptr
  call void @intrinsic_vec_push(ptr %t767, ptr %t768)
  br label %then57_end
then57_end:
  br label %endif57
else57:
  br label %else57_end
else57_end:
  br label %endif57
endif57:
  %t769 = phi i64 [ 0, %then57_end ], [ 0, %else57_end ]
  br label %then56_end
then56_end:
  br label %endif56
else56:
  br label %else56_end
else56_end:
  br label %endif56
endif56:
  %t770 = phi i64 [ %t769, %then56_end ], [ 0, %else56_end ]
  %t771 = load i64, ptr %local.i.619
  %t772 = add i64 %t771, 2
  store i64 %t772, ptr %local.i.619
  br label %then55_end
then55_end:
  br label %endif55
else55:
  %t773 = load i64, ptr %local.i.619
  %t774 = add i64 %t773, 1
  store i64 %t774, ptr %local.i.619
  br label %else55_end
else55_end:
  br label %endif55
endif55:
  %t775 = phi i64 [ 0, %then55_end ], [ 0, %else55_end ]
  br label %then54_end
then54_end:
  br label %endif54
else54:
  %t776 = load i64, ptr %local.i.619
  %t777 = add i64 %t776, 1
  store i64 %t777, ptr %local.i.619
  br label %else54_end
else54_end:
  br label %endif54
endif54:
  %t778 = phi i64 [ %t775, %then54_end ], [ 0, %else54_end ]
  br label %else50_end
else50_end:
  br label %endif50
endif50:
  %t779 = phi i64 [ %t718, %then50_end ], [ %t778, %else50_end ]
  br label %loop49
endloop49:
  %t780 = load i64, ptr %local.deps.614
  ret i64 %t780
}

define i64 @"get_directory"(i64 %path) {
entry:
  %local.len.781 = alloca i64
  %local.i.782 = alloca i64
  %local.c.783 = alloca i64
  %local.path = alloca i64
  store i64 %path, ptr %local.path
  %t784 = load i64, ptr %local.path
  %t785 = inttoptr i64 %t784 to ptr
  %t786 = call i64 @intrinsic_string_len(ptr %t785)
  store i64 %t786, ptr %local.len.781
  %t787 = load i64, ptr %local.len.781
  %t788 = sub i64 %t787, 1
  store i64 %t788, ptr %local.i.782
  br label %loop58
loop58:
  %t789 = load i64, ptr %local.i.782
  %t790 = icmp sge i64 %t789, 0
  %t791 = zext i1 %t790 to i64
  %t792 = icmp ne i64 %t791, 0
  br i1 %t792, label %body58, label %endloop58
body58:
  %t793 = load i64, ptr %local.path
  %t794 = load i64, ptr %local.i.782
  %t795 = inttoptr i64 %t793 to ptr
  %t796 = call i64 @intrinsic_string_char_at(ptr %t795, i64 %t794)
  store i64 %t796, ptr %local.c.783
  %t797 = load i64, ptr %local.c.783
  %t798 = icmp eq i64 %t797, 47
  %t799 = zext i1 %t798 to i64
  %t800 = icmp ne i64 %t799, 0
  br i1 %t800, label %then59, label %else59
then59:
  %t801 = load i64, ptr %local.path
  %t802 = load i64, ptr %local.i.782
  %t803 = add i64 %t802, 1
  %t804 = inttoptr i64 %t801 to ptr
  %t805 = call ptr @intrinsic_string_slice(ptr %t804, i64 0, i64 %t803)
  %t806 = ptrtoint ptr %t805 to i64
  ret i64 %t806
  br label %then59_end
then59_end:
  br label %endif59
else59:
  br label %else59_end
else59_end:
  br label %endif59
endif59:
  %t807 = phi i64 [ 0, %then59_end ], [ 0, %else59_end ]
  %t808 = load i64, ptr %local.i.782
  %t809 = sub i64 %t808, 1
  store i64 %t809, ptr %local.i.782
  br label %loop58
endloop58:
  %t810 = call ptr @intrinsic_string_new(ptr @.str.main.49)
  %t811 = ptrtoint ptr %t810 to i64
  ret i64 %t811
}

define i64 @"get_basename"(i64 %path) {
entry:
  %local.len.812 = alloca i64
  %local.i.813 = alloca i64
  %local.c.814 = alloca i64
  %local.path = alloca i64
  store i64 %path, ptr %local.path
  %t815 = load i64, ptr %local.path
  %t816 = inttoptr i64 %t815 to ptr
  %t817 = call i64 @intrinsic_string_len(ptr %t816)
  store i64 %t817, ptr %local.len.812
  %t818 = load i64, ptr %local.len.812
  %t819 = sub i64 %t818, 1
  store i64 %t819, ptr %local.i.813
  br label %loop60
loop60:
  %t820 = load i64, ptr %local.i.813
  %t821 = icmp sge i64 %t820, 0
  %t822 = zext i1 %t821 to i64
  %t823 = icmp ne i64 %t822, 0
  br i1 %t823, label %body60, label %endloop60
body60:
  %t824 = load i64, ptr %local.path
  %t825 = load i64, ptr %local.i.813
  %t826 = inttoptr i64 %t824 to ptr
  %t827 = call i64 @intrinsic_string_char_at(ptr %t826, i64 %t825)
  store i64 %t827, ptr %local.c.814
  %t828 = load i64, ptr %local.c.814
  %t829 = icmp eq i64 %t828, 47
  %t830 = zext i1 %t829 to i64
  %t831 = icmp ne i64 %t830, 0
  br i1 %t831, label %then61, label %else61
then61:
  %t832 = load i64, ptr %local.path
  %t833 = load i64, ptr %local.i.813
  %t834 = add i64 %t833, 1
  %t835 = load i64, ptr %local.len.812
  %t836 = inttoptr i64 %t832 to ptr
  %t837 = call ptr @intrinsic_string_slice(ptr %t836, i64 %t834, i64 %t835)
  %t838 = ptrtoint ptr %t837 to i64
  ret i64 %t838
  br label %then61_end
then61_end:
  br label %endif61
else61:
  br label %else61_end
else61_end:
  br label %endif61
endif61:
  %t839 = phi i64 [ 0, %then61_end ], [ 0, %else61_end ]
  %t840 = load i64, ptr %local.i.813
  %t841 = sub i64 %t840, 1
  store i64 %t841, ptr %local.i.813
  br label %loop60
endloop60:
  %t842 = load i64, ptr %local.path
  ret i64 %t842
}

define i64 @"collect_all_dependencies"(i64 %source_path) {
entry:
  %local.result.843 = alloca i64
  %local.visited.844 = alloca i64
  %local.source_path = alloca i64
  store i64 %source_path, ptr %local.source_path
  %t845 = call ptr @intrinsic_vec_new()
  %t846 = ptrtoint ptr %t845 to i64
  store i64 %t846, ptr %local.result.843
  %t847 = call ptr @intrinsic_vec_new()
  %t848 = ptrtoint ptr %t847 to i64
  store i64 %t848, ptr %local.visited.844
  %t849 = load i64, ptr %local.source_path
  %t850 = load i64, ptr %local.result.843
  %t851 = load i64, ptr %local.visited.844
  %t852 = call i64 @"collect_deps_recursive"(i64 %t849, i64 %t850, i64 %t851)
  %t853 = load i64, ptr %local.result.843
  ret i64 %t853
}

define i64 @"collect_deps_recursive"(i64 %path, i64 %result, i64 %visited) {
entry:
  %local.deps.854 = alloca i64
  %local.n.855 = alloca i64
  %local.i.856 = alloca i64
  %local.dep.857 = alloca i64
  %local.path = alloca i64
  store i64 %path, ptr %local.path
  %local.result = alloca i64
  store i64 %result, ptr %local.result
  %local.visited = alloca i64
  store i64 %visited, ptr %local.visited
  %t858 = load i64, ptr %local.visited
  %t859 = load i64, ptr %local.path
  %t860 = call i64 @"vec_contains_string"(i64 %t858, i64 %t859)
  %t861 = icmp eq i64 %t860, 1
  %t862 = zext i1 %t861 to i64
  %t863 = icmp ne i64 %t862, 0
  br i1 %t863, label %then62, label %else62
then62:
  ret i64 0
  br label %then62_end
then62_end:
  br label %endif62
else62:
  br label %else62_end
else62_end:
  br label %endif62
endif62:
  %t864 = phi i64 [ 0, %then62_end ], [ 0, %else62_end ]
  %t865 = load i64, ptr %local.visited
  %t866 = load i64, ptr %local.path
  %t867 = inttoptr i64 %t865 to ptr
  %t868 = inttoptr i64 %t866 to ptr
  call void @intrinsic_vec_push(ptr %t867, ptr %t868)
  %t869 = load i64, ptr %local.path
  %t870 = call i64 @"extract_dependencies"(i64 %t869)
  store i64 %t870, ptr %local.deps.854
  %t871 = load i64, ptr %local.deps.854
  %t872 = inttoptr i64 %t871 to ptr
  %t873 = call i64 @intrinsic_vec_len(ptr %t872)
  store i64 %t873, ptr %local.n.855
  store i64 0, ptr %local.i.856
  br label %loop63
loop63:
  %t874 = load i64, ptr %local.i.856
  %t875 = load i64, ptr %local.n.855
  %t876 = icmp slt i64 %t874, %t875
  %t877 = zext i1 %t876 to i64
  %t878 = icmp ne i64 %t877, 0
  br i1 %t878, label %body63, label %endloop63
body63:
  %t879 = load i64, ptr %local.deps.854
  %t880 = load i64, ptr %local.i.856
  %t881 = inttoptr i64 %t879 to ptr
  %t882 = call ptr @intrinsic_vec_get(ptr %t881, i64 %t880)
  %t883 = ptrtoint ptr %t882 to i64
  store i64 %t883, ptr %local.dep.857
  %t884 = load i64, ptr %local.dep.857
  %t885 = load i64, ptr %local.result
  %t886 = load i64, ptr %local.visited
  %t887 = call i64 @"collect_deps_recursive"(i64 %t884, i64 %t885, i64 %t886)
  %t888 = load i64, ptr %local.i.856
  %t889 = add i64 %t888, 1
  store i64 %t889, ptr %local.i.856
  br label %loop63
endloop63:
  %t890 = load i64, ptr %local.result
  %t891 = load i64, ptr %local.path
  %t892 = call i64 @"vec_contains_string"(i64 %t890, i64 %t891)
  %t893 = icmp eq i64 %t892, 0
  %t894 = zext i1 %t893 to i64
  %t895 = icmp ne i64 %t894, 0
  br i1 %t895, label %then64, label %else64
then64:
  %t896 = load i64, ptr %local.result
  %t897 = load i64, ptr %local.path
  %t898 = inttoptr i64 %t896 to ptr
  %t899 = inttoptr i64 %t897 to ptr
  call void @intrinsic_vec_push(ptr %t898, ptr %t899)
  br label %then64_end
then64_end:
  br label %endif64
else64:
  br label %else64_end
else64_end:
  br label %endif64
endif64:
  %t900 = phi i64 [ 0, %then64_end ], [ 0, %else64_end ]
  ret i64 0
}

define i64 @"compile_with_dependencies"(i64 %path, i64 %output_path, i64 %verbose, i64 %emit_type) {
entry:
  %local.all_files.901 = alloca i64
  %local.n.902 = alloca i64
  %local.i.903 = alloca i64
  %local.dep_path.904 = alloca i64
  %local.result.905 = alloca i64
  %local.path = alloca i64
  store i64 %path, ptr %local.path
  %local.output_path = alloca i64
  store i64 %output_path, ptr %local.output_path
  %local.verbose = alloca i64
  store i64 %verbose, ptr %local.verbose
  %local.emit_type = alloca i64
  store i64 %emit_type, ptr %local.emit_type
  %t906 = load i64, ptr %local.path
  %t907 = call i64 @"collect_all_dependencies"(i64 %t906)
  store i64 %t907, ptr %local.all_files.901
  %t908 = load i64, ptr %local.all_files.901
  %t909 = inttoptr i64 %t908 to ptr
  %t910 = call i64 @intrinsic_vec_len(ptr %t909)
  store i64 %t910, ptr %local.n.902
  %t911 = load i64, ptr %local.verbose
  %t912 = icmp eq i64 %t911, 1
  %t913 = zext i1 %t912 to i64
  %t914 = icmp ne i64 %t913, 0
  br i1 %t914, label %then65, label %else65
then65:
  %t915 = load i64, ptr %local.n.902
  %t916 = icmp sgt i64 %t915, 1
  %t917 = zext i1 %t916 to i64
  %t918 = icmp ne i64 %t917, 0
  br i1 %t918, label %then66, label %else66
then66:
  %t919 = call ptr @intrinsic_string_new(ptr @.str.main.50)
  %t920 = ptrtoint ptr %t919 to i64
  %t921 = load i64, ptr %local.n.902
  %t922 = call ptr @intrinsic_int_to_string(i64 %t921)
  %t923 = ptrtoint ptr %t922 to i64
  %t924 = call ptr @intrinsic_string_new(ptr @.str.main.51)
  %t925 = ptrtoint ptr %t924 to i64
  %t926 = inttoptr i64 %t923 to ptr
  %t927 = inttoptr i64 %t925 to ptr
  %t928 = call ptr @intrinsic_string_concat(ptr %t926, ptr %t927)
  %t929 = ptrtoint ptr %t928 to i64
  %t930 = inttoptr i64 %t920 to ptr
  %t931 = inttoptr i64 %t929 to ptr
  %t932 = call ptr @intrinsic_string_concat(ptr %t930, ptr %t931)
  %t933 = ptrtoint ptr %t932 to i64
  %t934 = inttoptr i64 %t933 to ptr
  call void @intrinsic_println(ptr %t934)
  br label %then66_end
then66_end:
  br label %endif66
else66:
  br label %else66_end
else66_end:
  br label %endif66
endif66:
  %t935 = phi i64 [ 0, %then66_end ], [ 0, %else66_end ]
  br label %then65_end
then65_end:
  br label %endif65
else65:
  br label %else65_end
else65_end:
  br label %endif65
endif65:
  %t936 = phi i64 [ %t935, %then65_end ], [ 0, %else65_end ]
  store i64 0, ptr %local.i.903
  br label %loop67
loop67:
  %t937 = load i64, ptr %local.i.903
  %t938 = load i64, ptr %local.n.902
  %t939 = sub i64 %t938, 1
  %t940 = icmp slt i64 %t937, %t939
  %t941 = zext i1 %t940 to i64
  %t942 = icmp ne i64 %t941, 0
  br i1 %t942, label %body67, label %endloop67
body67:
  %t943 = load i64, ptr %local.all_files.901
  %t944 = load i64, ptr %local.i.903
  %t945 = inttoptr i64 %t943 to ptr
  %t946 = call ptr @intrinsic_vec_get(ptr %t945, i64 %t944)
  %t947 = ptrtoint ptr %t946 to i64
  store i64 %t947, ptr %local.dep_path.904
  %t948 = load i64, ptr %local.verbose
  %t949 = icmp eq i64 %t948, 1
  %t950 = zext i1 %t949 to i64
  %t951 = icmp ne i64 %t950, 0
  br i1 %t951, label %then68, label %else68
then68:
  %t952 = call ptr @intrinsic_string_new(ptr @.str.main.52)
  %t953 = ptrtoint ptr %t952 to i64
  %t954 = load i64, ptr %local.dep_path.904
  %t955 = inttoptr i64 %t953 to ptr
  %t956 = inttoptr i64 %t954 to ptr
  %t957 = call ptr @intrinsic_string_concat(ptr %t955, ptr %t956)
  %t958 = ptrtoint ptr %t957 to i64
  %t959 = inttoptr i64 %t958 to ptr
  call void @intrinsic_println(ptr %t959)
  br label %then68_end
then68_end:
  br label %endif68
else68:
  br label %else68_end
else68_end:
  br label %endif68
endif68:
  %t960 = phi i64 [ 0, %then68_end ], [ 0, %else68_end ]
  %t961 = load i64, ptr %local.dep_path.904
  %t962 = call i64 @"compile_file"(i64 %t961, i64 0, i64 0, i64 0)
  store i64 %t962, ptr %local.result.905
  %t963 = load i64, ptr %local.result.905
  %t964 = icmp ne i64 %t963, 0
  %t965 = zext i1 %t964 to i64
  %t966 = icmp ne i64 %t965, 0
  br i1 %t966, label %then69, label %else69
then69:
  %t967 = call ptr @intrinsic_string_new(ptr @.str.main.53)
  %t968 = ptrtoint ptr %t967 to i64
  %t969 = load i64, ptr %local.dep_path.904
  %t970 = inttoptr i64 %t968 to ptr
  %t971 = inttoptr i64 %t969 to ptr
  %t972 = call ptr @intrinsic_string_concat(ptr %t970, ptr %t971)
  %t973 = ptrtoint ptr %t972 to i64
  %t974 = inttoptr i64 %t973 to ptr
  call void @intrinsic_println(ptr %t974)
  ret i64 1
  br label %then69_end
then69_end:
  br label %endif69
else69:
  br label %else69_end
else69_end:
  br label %endif69
endif69:
  %t975 = phi i64 [ 0, %then69_end ], [ 0, %else69_end ]
  %t976 = load i64, ptr %local.i.903
  %t977 = add i64 %t976, 1
  store i64 %t977, ptr %local.i.903
  br label %loop67
endloop67:
  %t978 = load i64, ptr %local.path
  %t979 = load i64, ptr %local.output_path
  %t980 = load i64, ptr %local.verbose
  %t981 = load i64, ptr %local.emit_type
  %t982 = call i64 @"compile_file"(i64 %t978, i64 %t979, i64 %t980, i64 %t981)
  ret i64 %t982
}

define i64 @"is_up_to_date_with_deps"(i64 %source_path, i64 %output_path) {
entry:
  %local.out_mtime.983 = alloca i64
  %local.src_mtime.984 = alloca i64
  %local.deps.985 = alloca i64
  %local.n.986 = alloca i64
  %local.i.987 = alloca i64
  %local.dep_path.988 = alloca i64
  %local.dep_mtime.989 = alloca i64
  %local.source_path = alloca i64
  store i64 %source_path, ptr %local.source_path
  %local.output_path = alloca i64
  store i64 %output_path, ptr %local.output_path
  %t990 = load i64, ptr %local.output_path
  %t991 = inttoptr i64 %t990 to ptr
  %t992 = call i64 @intrinsic_file_exists(ptr %t991)
  %t993 = icmp eq i64 %t992, 0
  %t994 = zext i1 %t993 to i64
  %t995 = icmp ne i64 %t994, 0
  br i1 %t995, label %then70, label %else70
then70:
  ret i64 0
  br label %then70_end
then70_end:
  br label %endif70
else70:
  br label %else70_end
else70_end:
  br label %endif70
endif70:
  %t996 = phi i64 [ 0, %then70_end ], [ 0, %else70_end ]
  %t997 = load i64, ptr %local.output_path
  %t998 = inttoptr i64 %t997 to ptr
  %t999 = call i64 @intrinsic_file_mtime(ptr %t998)
  store i64 %t999, ptr %local.out_mtime.983
  %t1000 = load i64, ptr %local.source_path
  %t1001 = inttoptr i64 %t1000 to ptr
  %t1002 = call i64 @intrinsic_file_mtime(ptr %t1001)
  store i64 %t1002, ptr %local.src_mtime.984
  %t1003 = load i64, ptr %local.src_mtime.984
  %t1004 = load i64, ptr %local.out_mtime.983
  %t1005 = icmp sgt i64 %t1003, %t1004
  %t1006 = zext i1 %t1005 to i64
  %t1007 = icmp ne i64 %t1006, 0
  br i1 %t1007, label %then71, label %else71
then71:
  ret i64 0
  br label %then71_end
then71_end:
  br label %endif71
else71:
  br label %else71_end
else71_end:
  br label %endif71
endif71:
  %t1008 = phi i64 [ 0, %then71_end ], [ 0, %else71_end ]
  %t1009 = load i64, ptr %local.source_path
  %t1010 = call i64 @"extract_dependencies"(i64 %t1009)
  store i64 %t1010, ptr %local.deps.985
  %t1011 = load i64, ptr %local.deps.985
  %t1012 = inttoptr i64 %t1011 to ptr
  %t1013 = call i64 @intrinsic_vec_len(ptr %t1012)
  store i64 %t1013, ptr %local.n.986
  store i64 0, ptr %local.i.987
  br label %loop72
loop72:
  %t1014 = load i64, ptr %local.i.987
  %t1015 = load i64, ptr %local.n.986
  %t1016 = icmp slt i64 %t1014, %t1015
  %t1017 = zext i1 %t1016 to i64
  %t1018 = icmp ne i64 %t1017, 0
  br i1 %t1018, label %body72, label %endloop72
body72:
  %t1019 = load i64, ptr %local.deps.985
  %t1020 = load i64, ptr %local.i.987
  %t1021 = inttoptr i64 %t1019 to ptr
  %t1022 = call ptr @intrinsic_vec_get(ptr %t1021, i64 %t1020)
  %t1023 = ptrtoint ptr %t1022 to i64
  store i64 %t1023, ptr %local.dep_path.988
  %t1024 = load i64, ptr %local.dep_path.988
  %t1025 = inttoptr i64 %t1024 to ptr
  %t1026 = call i64 @intrinsic_file_mtime(ptr %t1025)
  store i64 %t1026, ptr %local.dep_mtime.989
  %t1027 = load i64, ptr %local.dep_mtime.989
  %t1028 = load i64, ptr %local.out_mtime.983
  %t1029 = icmp sgt i64 %t1027, %t1028
  %t1030 = zext i1 %t1029 to i64
  %t1031 = icmp ne i64 %t1030, 0
  br i1 %t1031, label %then73, label %else73
then73:
  ret i64 0
  br label %then73_end
then73_end:
  br label %endif73
else73:
  br label %else73_end
else73_end:
  br label %endif73
endif73:
  %t1032 = phi i64 [ 0, %then73_end ], [ 0, %else73_end ]
  %t1033 = load i64, ptr %local.dep_path.988
  %t1034 = load i64, ptr %local.output_path
  %t1035 = call i64 @"is_up_to_date_with_deps"(i64 %t1033, i64 %t1034)
  %t1036 = icmp eq i64 %t1035, 0
  %t1037 = zext i1 %t1036 to i64
  %t1038 = icmp ne i64 %t1037, 0
  br i1 %t1038, label %then74, label %else74
then74:
  ret i64 0
  br label %then74_end
then74_end:
  br label %endif74
else74:
  br label %else74_end
else74_end:
  br label %endif74
endif74:
  %t1039 = phi i64 [ 0, %then74_end ], [ 0, %else74_end ]
  %t1040 = load i64, ptr %local.i.987
  %t1041 = add i64 %t1040, 1
  store i64 %t1041, ptr %local.i.987
  br label %loop72
endloop72:
  ret i64 1
}

define i64 @"compile_file"(i64 %path, i64 %output_path, i64 %verbose, i64 %emit_type) {
entry:
  %local.path = alloca i64
  store i64 %path, ptr %local.path
  %local.output_path = alloca i64
  store i64 %output_path, ptr %local.output_path
  %local.verbose = alloca i64
  store i64 %verbose, ptr %local.verbose
  %local.emit_type = alloca i64
  store i64 %emit_type, ptr %local.emit_type
  %t1042 = load i64, ptr %local.path
  %t1043 = load i64, ptr %local.output_path
  %t1044 = load i64, ptr %local.verbose
  %t1045 = load i64, ptr %local.emit_type
  %t1046 = call i64 @"compile_file_full"(i64 %t1042, i64 %t1043, i64 %t1044, i64 %t1045, i64 0, i64 0)
  ret i64 %t1046
}

define i64 @"compile_file_force"(i64 %path, i64 %output_path, i64 %verbose, i64 %emit_type, i64 %force) {
entry:
  %local.path = alloca i64
  store i64 %path, ptr %local.path
  %local.output_path = alloca i64
  store i64 %output_path, ptr %local.output_path
  %local.verbose = alloca i64
  store i64 %verbose, ptr %local.verbose
  %local.emit_type = alloca i64
  store i64 %emit_type, ptr %local.emit_type
  %local.force = alloca i64
  store i64 %force, ptr %local.force
  %t1047 = load i64, ptr %local.path
  %t1048 = load i64, ptr %local.output_path
  %t1049 = load i64, ptr %local.verbose
  %t1050 = load i64, ptr %local.emit_type
  %t1051 = load i64, ptr %local.force
  %t1052 = call i64 @"compile_file_full"(i64 %t1047, i64 %t1048, i64 %t1049, i64 %t1050, i64 %t1051, i64 0)
  ret i64 %t1052
}

define i64 @"compile_file_full"(i64 %path, i64 %output_path, i64 %verbose, i64 %emit_type, i64 %force, i64 %debug_info) {
entry:
  %local.expected_out.1053 = alloca i64
  %local.source.1054 = alloca i64
  %local.tokens.1055 = alloca i64
  %local.parse_result.1056 = alloca i64
  %local.items.1057 = alloca i64
  %local.parse_errors.1058 = alloca i64
  %local.module_name.1059 = alloca i64
  %local.output.1060 = alloca i64
  %local.dir.1061 = alloca i64
  %local.filename.1062 = alloca i64
  %local.ll_path.1063 = alloca i64
  %local.out_path.1064 = alloca i64
  %local.asm_path.1065 = alloca i64
  %local.cmd.1066 = alloca i64
  %local.result.1067 = alloca i64
  %local.obj_path.1068 = alloca i64
  %local.cmd.1069 = alloca i64
  %local.result.1070 = alloca i64
  %local.exe_path.1071 = alloca i64
  %local.obj_path.1072 = alloca i64
  %local.cmd.1073 = alloca i64
  %local.result.1074 = alloca i64
  %local.dylib_path.1075 = alloca i64
  %local.obj_path.1076 = alloca i64
  %local.cmd.1077 = alloca i64
  %local.result.1078 = alloca i64
  %local.path = alloca i64
  store i64 %path, ptr %local.path
  %local.output_path = alloca i64
  store i64 %output_path, ptr %local.output_path
  %local.verbose = alloca i64
  store i64 %verbose, ptr %local.verbose
  %local.emit_type = alloca i64
  store i64 %emit_type, ptr %local.emit_type
  %local.force = alloca i64
  store i64 %force, ptr %local.force
  %local.debug_info = alloca i64
  store i64 %debug_info, ptr %local.debug_info
  %t1079 = load i64, ptr %local.output_path
  store i64 %t1079, ptr %local.expected_out.1053
  %t1080 = load i64, ptr %local.expected_out.1053
  %t1081 = icmp eq i64 %t1080, 0
  %t1082 = zext i1 %t1081 to i64
  %t1083 = icmp ne i64 %t1082, 0
  br i1 %t1083, label %then75, label %else75
then75:
  %t1084 = load i64, ptr %local.emit_type
  %t1085 = icmp eq i64 %t1084, 0
  %t1086 = zext i1 %t1085 to i64
  %t1087 = icmp ne i64 %t1086, 0
  br i1 %t1087, label %then76, label %else76
then76:
  %t1088 = load i64, ptr %local.path
  %t1089 = call ptr @intrinsic_string_new(ptr @.str.main.54)
  %t1090 = ptrtoint ptr %t1089 to i64
  %t1091 = call i64 @"string_replace_ext"(i64 %t1088, i64 %t1090)
  store i64 %t1091, ptr %local.expected_out.1053
  br label %then76_end
then76_end:
  br label %endif76
else76:
  %t1092 = load i64, ptr %local.emit_type
  %t1093 = icmp eq i64 %t1092, 1
  %t1094 = zext i1 %t1093 to i64
  %t1095 = icmp ne i64 %t1094, 0
  br i1 %t1095, label %then77, label %else77
then77:
  %t1096 = load i64, ptr %local.path
  %t1097 = call ptr @intrinsic_string_new(ptr @.str.main.55)
  %t1098 = ptrtoint ptr %t1097 to i64
  %t1099 = call i64 @"string_replace_ext"(i64 %t1096, i64 %t1098)
  store i64 %t1099, ptr %local.expected_out.1053
  br label %then77_end
then77_end:
  br label %endif77
else77:
  %t1100 = load i64, ptr %local.emit_type
  %t1101 = icmp eq i64 %t1100, 2
  %t1102 = zext i1 %t1101 to i64
  %t1103 = icmp ne i64 %t1102, 0
  br i1 %t1103, label %then78, label %else78
then78:
  %t1104 = load i64, ptr %local.path
  %t1105 = call ptr @intrinsic_string_new(ptr @.str.main.56)
  %t1106 = ptrtoint ptr %t1105 to i64
  %t1107 = call i64 @"string_replace_ext"(i64 %t1104, i64 %t1106)
  store i64 %t1107, ptr %local.expected_out.1053
  br label %then78_end
then78_end:
  br label %endif78
else78:
  %t1108 = load i64, ptr %local.emit_type
  %t1109 = icmp eq i64 %t1108, 3
  %t1110 = zext i1 %t1109 to i64
  %t1111 = icmp ne i64 %t1110, 0
  br i1 %t1111, label %then79, label %else79
then79:
  %t1112 = load i64, ptr %local.path
  %t1113 = call ptr @intrinsic_string_new(ptr @.str.main.57)
  %t1114 = ptrtoint ptr %t1113 to i64
  %t1115 = call i64 @"string_replace_ext"(i64 %t1112, i64 %t1114)
  store i64 %t1115, ptr %local.expected_out.1053
  br label %then79_end
then79_end:
  br label %endif79
else79:
  %t1116 = load i64, ptr %local.emit_type
  %t1117 = icmp eq i64 %t1116, 4
  %t1118 = zext i1 %t1117 to i64
  %t1119 = icmp ne i64 %t1118, 0
  br i1 %t1119, label %then80, label %else80
then80:
  %t1120 = load i64, ptr %local.path
  %t1121 = call ptr @intrinsic_string_new(ptr @.str.main.58)
  %t1122 = ptrtoint ptr %t1121 to i64
  %t1123 = call i64 @"string_replace_ext"(i64 %t1120, i64 %t1122)
  store i64 %t1123, ptr %local.expected_out.1053
  br label %then80_end
then80_end:
  br label %endif80
else80:
  br label %else80_end
else80_end:
  br label %endif80
endif80:
  %t1124 = phi i64 [ 0, %then80_end ], [ 0, %else80_end ]
  br label %else79_end
else79_end:
  br label %endif79
endif79:
  %t1125 = phi i64 [ 0, %then79_end ], [ %t1124, %else79_end ]
  br label %else78_end
else78_end:
  br label %endif78
endif78:
  %t1126 = phi i64 [ 0, %then78_end ], [ %t1125, %else78_end ]
  br label %else77_end
else77_end:
  br label %endif77
endif77:
  %t1127 = phi i64 [ 0, %then77_end ], [ %t1126, %else77_end ]
  br label %else76_end
else76_end:
  br label %endif76
endif76:
  %t1128 = phi i64 [ 0, %then76_end ], [ %t1127, %else76_end ]
  br label %then75_end
then75_end:
  br label %endif75
else75:
  br label %else75_end
else75_end:
  br label %endif75
endif75:
  %t1129 = phi i64 [ %t1128, %then75_end ], [ 0, %else75_end ]
  %t1130 = load i64, ptr %local.force
  %t1131 = icmp eq i64 %t1130, 0
  %t1132 = zext i1 %t1131 to i64
  %t1133 = icmp ne i64 %t1132, 0
  br i1 %t1133, label %then81, label %else81
then81:
  %t1134 = load i64, ptr %local.path
  %t1135 = load i64, ptr %local.expected_out.1053
  %t1136 = call i64 @"is_up_to_date_with_deps"(i64 %t1134, i64 %t1135)
  %t1137 = icmp eq i64 %t1136, 1
  %t1138 = zext i1 %t1137 to i64
  %t1139 = icmp ne i64 %t1138, 0
  br i1 %t1139, label %then82, label %else82
then82:
  %t1140 = load i64, ptr %local.verbose
  %t1141 = icmp eq i64 %t1140, 1
  %t1142 = zext i1 %t1141 to i64
  %t1143 = icmp ne i64 %t1142, 0
  br i1 %t1143, label %then83, label %else83
then83:
  %t1144 = call ptr @intrinsic_string_new(ptr @.str.main.59)
  %t1145 = ptrtoint ptr %t1144 to i64
  %t1146 = load i64, ptr %local.path
  %t1147 = inttoptr i64 %t1145 to ptr
  %t1148 = inttoptr i64 %t1146 to ptr
  %t1149 = call ptr @intrinsic_string_concat(ptr %t1147, ptr %t1148)
  %t1150 = ptrtoint ptr %t1149 to i64
  %t1151 = inttoptr i64 %t1150 to ptr
  call void @intrinsic_println(ptr %t1151)
  br label %then83_end
then83_end:
  br label %endif83
else83:
  br label %else83_end
else83_end:
  br label %endif83
endif83:
  %t1152 = phi i64 [ 0, %then83_end ], [ 0, %else83_end ]
  ret i64 0
  br label %then82_end
then82_end:
  br label %endif82
else82:
  br label %else82_end
else82_end:
  br label %endif82
endif82:
  %t1153 = phi i64 [ 0, %then82_end ], [ 0, %else82_end ]
  br label %then81_end
then81_end:
  br label %endif81
else81:
  br label %else81_end
else81_end:
  br label %endif81
endif81:
  %t1154 = phi i64 [ %t1153, %then81_end ], [ 0, %else81_end ]
  %t1155 = load i64, ptr %local.verbose
  %t1156 = icmp eq i64 %t1155, 1
  %t1157 = zext i1 %t1156 to i64
  %t1158 = icmp ne i64 %t1157, 0
  br i1 %t1158, label %then84, label %else84
then84:
  %t1159 = call ptr @intrinsic_string_new(ptr @.str.main.60)
  %t1160 = ptrtoint ptr %t1159 to i64
  %t1161 = load i64, ptr %local.path
  %t1162 = inttoptr i64 %t1160 to ptr
  %t1163 = inttoptr i64 %t1161 to ptr
  %t1164 = call ptr @intrinsic_string_concat(ptr %t1162, ptr %t1163)
  %t1165 = ptrtoint ptr %t1164 to i64
  %t1166 = inttoptr i64 %t1165 to ptr
  call void @intrinsic_println(ptr %t1166)
  br label %then84_end
then84_end:
  br label %endif84
else84:
  br label %else84_end
else84_end:
  br label %endif84
endif84:
  %t1167 = phi i64 [ 0, %then84_end ], [ 0, %else84_end ]
  %t1168 = load i64, ptr %local.path
  %t1169 = inttoptr i64 %t1168 to ptr
  %t1170 = call ptr @intrinsic_read_file(ptr %t1169)
  %t1171 = ptrtoint ptr %t1170 to i64
  store i64 %t1171, ptr %local.source.1054
  %t1172 = load i64, ptr %local.source.1054
  %t1173 = icmp eq i64 %t1172, 0
  %t1174 = zext i1 %t1173 to i64
  %t1175 = icmp ne i64 %t1174, 0
  br i1 %t1175, label %then85, label %else85
then85:
  %t1176 = call i64 @"E_CANNOT_READ_FILE"()
  %t1177 = call ptr @intrinsic_string_new(ptr @.str.main.61)
  %t1178 = ptrtoint ptr %t1177 to i64
  %t1179 = load i64, ptr %local.path
  %t1180 = inttoptr i64 %t1178 to ptr
  %t1181 = inttoptr i64 %t1179 to ptr
  %t1182 = call ptr @intrinsic_string_concat(ptr %t1180, ptr %t1181)
  %t1183 = ptrtoint ptr %t1182 to i64
  %t1184 = call i64 @"report_simple_error"(i64 %t1176, i64 %t1183)
  ret i64 1
  br label %then85_end
then85_end:
  br label %endif85
else85:
  br label %else85_end
else85_end:
  br label %endif85
endif85:
  %t1185 = phi i64 [ 0, %then85_end ], [ 0, %else85_end ]
  %t1186 = load i64, ptr %local.source.1054
  %t1187 = call i64 @"tokenize"(i64 %t1186)
  store i64 %t1187, ptr %local.tokens.1055
  %t1188 = load i64, ptr %local.tokens.1055
  %t1189 = load i64, ptr %local.source.1054
  %t1190 = load i64, ptr %local.path
  %t1191 = call i64 @"parse_program_with_source"(i64 %t1188, i64 %t1189, i64 %t1190)
  store i64 %t1191, ptr %local.parse_result.1056
  %t1192 = load i64, ptr %local.parse_result.1056
  %t1193 = call i64 @"parse_result_items"(i64 %t1192)
  store i64 %t1193, ptr %local.items.1057
  %t1194 = load i64, ptr %local.parse_result.1056
  %t1195 = call i64 @"parse_result_errors"(i64 %t1194)
  store i64 %t1195, ptr %local.parse_errors.1058
  %t1196 = load i64, ptr %local.parse_errors.1058
  %t1197 = icmp sgt i64 %t1196, 0
  %t1198 = zext i1 %t1197 to i64
  %t1199 = icmp ne i64 %t1198, 0
  br i1 %t1199, label %then86, label %else86
then86:
  %t1200 = load i64, ptr %local.parse_errors.1058
  %t1201 = call i64 @"print_error_summary"(i64 %t1200, i64 0)
  ret i64 1
  br label %then86_end
then86_end:
  br label %endif86
else86:
  br label %else86_end
else86_end:
  br label %endif86
endif86:
  %t1202 = phi i64 [ 0, %then86_end ], [ 0, %else86_end ]
  %t1203 = load i64, ptr %local.path
  %t1204 = call i64 @"get_module_name"(i64 %t1203)
  store i64 %t1204, ptr %local.module_name.1059
  store i64 0, ptr %local.output.1060
  %t1205 = load i64, ptr %local.debug_info
  %t1206 = icmp eq i64 %t1205, 1
  %t1207 = zext i1 %t1206 to i64
  %t1208 = icmp ne i64 %t1207, 0
  br i1 %t1208, label %then87, label %else87
then87:
  %t1209 = load i64, ptr %local.path
  %t1210 = call i64 @"get_directory"(i64 %t1209)
  store i64 %t1210, ptr %local.dir.1061
  %t1211 = load i64, ptr %local.dir.1061
  %t1212 = inttoptr i64 %t1211 to ptr
  %t1213 = call i64 @intrinsic_string_len(ptr %t1212)
  %t1214 = icmp eq i64 %t1213, 0
  %t1215 = zext i1 %t1214 to i64
  %t1216 = icmp ne i64 %t1215, 0
  br i1 %t1216, label %then88, label %else88
then88:
  %t1217 = call ptr @intrinsic_string_new(ptr @.str.main.62)
  %t1218 = ptrtoint ptr %t1217 to i64
  store i64 %t1218, ptr %local.dir.1061
  br label %then88_end
then88_end:
  br label %endif88
else88:
  br label %else88_end
else88_end:
  br label %endif88
endif88:
  %t1219 = phi i64 [ 0, %then88_end ], [ 0, %else88_end ]
  %t1220 = load i64, ptr %local.path
  %t1221 = call i64 @"get_basename"(i64 %t1220)
  store i64 %t1221, ptr %local.filename.1062
  %t1222 = load i64, ptr %local.items.1057
  %t1223 = load i64, ptr %local.module_name.1059
  %t1224 = load i64, ptr %local.filename.1062
  %t1225 = load i64, ptr %local.dir.1061
  %t1226 = call i64 @"gen_program_with_debug"(i64 %t1222, i64 %t1223, i64 %t1224, i64 %t1225)
  store i64 %t1226, ptr %local.output.1060
  br label %then87_end
then87_end:
  br label %endif87
else87:
  %t1227 = load i64, ptr %local.items.1057
  %t1228 = load i64, ptr %local.module_name.1059
  %t1229 = call i64 @"gen_program_with_module"(i64 %t1227, i64 %t1228)
  store i64 %t1229, ptr %local.output.1060
  br label %else87_end
else87_end:
  br label %endif87
endif87:
  %t1230 = phi i64 [ 0, %then87_end ], [ 0, %else87_end ]
  %t1231 = load i64, ptr %local.path
  %t1232 = call ptr @intrinsic_string_new(ptr @.str.main.63)
  %t1233 = ptrtoint ptr %t1232 to i64
  %t1234 = call i64 @"string_replace_ext"(i64 %t1231, i64 %t1233)
  store i64 %t1234, ptr %local.ll_path.1063
  %t1235 = load i64, ptr %local.output_path
  store i64 %t1235, ptr %local.out_path.1064
  %t1236 = load i64, ptr %local.emit_type
  %t1237 = icmp eq i64 %t1236, 0
  %t1238 = zext i1 %t1237 to i64
  %t1239 = icmp ne i64 %t1238, 0
  br i1 %t1239, label %then89, label %else89
then89:
  %t1240 = load i64, ptr %local.out_path.1064
  %t1241 = icmp eq i64 %t1240, 0
  %t1242 = zext i1 %t1241 to i64
  %t1243 = icmp ne i64 %t1242, 0
  br i1 %t1243, label %then90, label %else90
then90:
  %t1244 = load i64, ptr %local.ll_path.1063
  store i64 %t1244, ptr %local.out_path.1064
  br label %then90_end
then90_end:
  br label %endif90
else90:
  br label %else90_end
else90_end:
  br label %endif90
endif90:
  %t1245 = phi i64 [ 0, %then90_end ], [ 0, %else90_end ]
  %t1246 = load i64, ptr %local.out_path.1064
  %t1247 = load i64, ptr %local.output.1060
  %t1248 = inttoptr i64 %t1246 to ptr
  %t1249 = inttoptr i64 %t1247 to ptr
  call void @intrinsic_write_file(ptr %t1248, ptr %t1249)
  %t1250 = load i64, ptr %local.verbose
  %t1251 = icmp eq i64 %t1250, 1
  %t1252 = zext i1 %t1251 to i64
  %t1253 = icmp ne i64 %t1252, 0
  br i1 %t1253, label %then91, label %else91
then91:
  %t1254 = call ptr @intrinsic_string_new(ptr @.str.main.64)
  %t1255 = ptrtoint ptr %t1254 to i64
  %t1256 = load i64, ptr %local.out_path.1064
  %t1257 = inttoptr i64 %t1255 to ptr
  %t1258 = inttoptr i64 %t1256 to ptr
  %t1259 = call ptr @intrinsic_string_concat(ptr %t1257, ptr %t1258)
  %t1260 = ptrtoint ptr %t1259 to i64
  %t1261 = inttoptr i64 %t1260 to ptr
  call void @intrinsic_println(ptr %t1261)
  br label %then91_end
then91_end:
  br label %endif91
else91:
  br label %else91_end
else91_end:
  br label %endif91
endif91:
  %t1262 = phi i64 [ 0, %then91_end ], [ 0, %else91_end ]
  ret i64 0
  br label %then89_end
then89_end:
  br label %endif89
else89:
  br label %else89_end
else89_end:
  br label %endif89
endif89:
  %t1263 = phi i64 [ 0, %then89_end ], [ 0, %else89_end ]
  %t1264 = load i64, ptr %local.ll_path.1063
  %t1265 = load i64, ptr %local.output.1060
  %t1266 = inttoptr i64 %t1264 to ptr
  %t1267 = inttoptr i64 %t1265 to ptr
  call void @intrinsic_write_file(ptr %t1266, ptr %t1267)
  %t1268 = load i64, ptr %local.emit_type
  %t1269 = icmp eq i64 %t1268, 3
  %t1270 = zext i1 %t1269 to i64
  %t1271 = icmp ne i64 %t1270, 0
  br i1 %t1271, label %then92, label %else92
then92:
  %t1272 = load i64, ptr %local.out_path.1064
  store i64 %t1272, ptr %local.asm_path.1065
  %t1273 = load i64, ptr %local.asm_path.1065
  %t1274 = icmp eq i64 %t1273, 0
  %t1275 = zext i1 %t1274 to i64
  %t1276 = icmp ne i64 %t1275, 0
  br i1 %t1276, label %then93, label %else93
then93:
  %t1277 = load i64, ptr %local.path
  %t1278 = call ptr @intrinsic_string_new(ptr @.str.main.65)
  %t1279 = ptrtoint ptr %t1278 to i64
  %t1280 = call i64 @"string_replace_ext"(i64 %t1277, i64 %t1279)
  store i64 %t1280, ptr %local.asm_path.1065
  br label %then93_end
then93_end:
  br label %endif93
else93:
  br label %else93_end
else93_end:
  br label %endif93
endif93:
  %t1281 = phi i64 [ 0, %then93_end ], [ 0, %else93_end ]
  %t1282 = call ptr @intrinsic_string_new(ptr @.str.main.66)
  %t1283 = ptrtoint ptr %t1282 to i64
  %t1284 = load i64, ptr %local.asm_path.1065
  %t1285 = inttoptr i64 %t1283 to ptr
  %t1286 = inttoptr i64 %t1284 to ptr
  %t1287 = call ptr @intrinsic_string_concat(ptr %t1285, ptr %t1286)
  %t1288 = ptrtoint ptr %t1287 to i64
  store i64 %t1288, ptr %local.cmd.1066
  %t1289 = load i64, ptr %local.cmd.1066
  %t1290 = call ptr @intrinsic_string_new(ptr @.str.main.67)
  %t1291 = ptrtoint ptr %t1290 to i64
  %t1292 = inttoptr i64 %t1289 to ptr
  %t1293 = inttoptr i64 %t1291 to ptr
  %t1294 = call ptr @intrinsic_string_concat(ptr %t1292, ptr %t1293)
  %t1295 = ptrtoint ptr %t1294 to i64
  store i64 %t1295, ptr %local.cmd.1066
  %t1296 = load i64, ptr %local.cmd.1066
  %t1297 = load i64, ptr %local.ll_path.1063
  %t1298 = inttoptr i64 %t1296 to ptr
  %t1299 = inttoptr i64 %t1297 to ptr
  %t1300 = call ptr @intrinsic_string_concat(ptr %t1298, ptr %t1299)
  %t1301 = ptrtoint ptr %t1300 to i64
  store i64 %t1301, ptr %local.cmd.1066
  %t1302 = load i64, ptr %local.cmd.1066
  %t1303 = inttoptr i64 %t1302 to ptr
  %t1304 = call i64 @intrinsic_process_run(ptr %t1303)
  store i64 %t1304, ptr %local.result.1067
  %t1305 = load i64, ptr %local.result.1067
  %t1306 = icmp ne i64 %t1305, 0
  %t1307 = zext i1 %t1306 to i64
  %t1308 = icmp ne i64 %t1307, 0
  br i1 %t1308, label %then94, label %else94
then94:
  %t1309 = call ptr @intrinsic_string_new(ptr @.str.main.68)
  %t1310 = ptrtoint ptr %t1309 to i64
  %t1311 = inttoptr i64 %t1310 to ptr
  call void @intrinsic_println(ptr %t1311)
  ret i64 1
  br label %then94_end
then94_end:
  br label %endif94
else94:
  br label %else94_end
else94_end:
  br label %endif94
endif94:
  %t1312 = phi i64 [ 0, %then94_end ], [ 0, %else94_end ]
  %t1313 = load i64, ptr %local.verbose
  %t1314 = icmp eq i64 %t1313, 1
  %t1315 = zext i1 %t1314 to i64
  %t1316 = icmp ne i64 %t1315, 0
  br i1 %t1316, label %then95, label %else95
then95:
  %t1317 = call ptr @intrinsic_string_new(ptr @.str.main.69)
  %t1318 = ptrtoint ptr %t1317 to i64
  %t1319 = load i64, ptr %local.asm_path.1065
  %t1320 = inttoptr i64 %t1318 to ptr
  %t1321 = inttoptr i64 %t1319 to ptr
  %t1322 = call ptr @intrinsic_string_concat(ptr %t1320, ptr %t1321)
  %t1323 = ptrtoint ptr %t1322 to i64
  %t1324 = inttoptr i64 %t1323 to ptr
  call void @intrinsic_println(ptr %t1324)
  br label %then95_end
then95_end:
  br label %endif95
else95:
  br label %else95_end
else95_end:
  br label %endif95
endif95:
  %t1325 = phi i64 [ 0, %then95_end ], [ 0, %else95_end ]
  ret i64 0
  br label %then92_end
then92_end:
  br label %endif92
else92:
  br label %else92_end
else92_end:
  br label %endif92
endif92:
  %t1326 = phi i64 [ 0, %then92_end ], [ 0, %else92_end ]
  %t1327 = load i64, ptr %local.emit_type
  %t1328 = icmp eq i64 %t1327, 1
  %t1329 = zext i1 %t1328 to i64
  %t1330 = icmp ne i64 %t1329, 0
  br i1 %t1330, label %then96, label %else96
then96:
  %t1331 = load i64, ptr %local.out_path.1064
  store i64 %t1331, ptr %local.obj_path.1068
  %t1332 = load i64, ptr %local.obj_path.1068
  %t1333 = icmp eq i64 %t1332, 0
  %t1334 = zext i1 %t1333 to i64
  %t1335 = icmp ne i64 %t1334, 0
  br i1 %t1335, label %then97, label %else97
then97:
  %t1336 = load i64, ptr %local.path
  %t1337 = call ptr @intrinsic_string_new(ptr @.str.main.70)
  %t1338 = ptrtoint ptr %t1337 to i64
  %t1339 = call i64 @"string_replace_ext"(i64 %t1336, i64 %t1338)
  store i64 %t1339, ptr %local.obj_path.1068
  br label %then97_end
then97_end:
  br label %endif97
else97:
  br label %else97_end
else97_end:
  br label %endif97
endif97:
  %t1340 = phi i64 [ 0, %then97_end ], [ 0, %else97_end ]
  %t1341 = call ptr @intrinsic_string_new(ptr @.str.main.71)
  %t1342 = ptrtoint ptr %t1341 to i64
  %t1343 = load i64, ptr %local.obj_path.1068
  %t1344 = inttoptr i64 %t1342 to ptr
  %t1345 = inttoptr i64 %t1343 to ptr
  %t1346 = call ptr @intrinsic_string_concat(ptr %t1344, ptr %t1345)
  %t1347 = ptrtoint ptr %t1346 to i64
  store i64 %t1347, ptr %local.cmd.1069
  %t1348 = load i64, ptr %local.cmd.1069
  %t1349 = call ptr @intrinsic_string_new(ptr @.str.main.72)
  %t1350 = ptrtoint ptr %t1349 to i64
  %t1351 = inttoptr i64 %t1348 to ptr
  %t1352 = inttoptr i64 %t1350 to ptr
  %t1353 = call ptr @intrinsic_string_concat(ptr %t1351, ptr %t1352)
  %t1354 = ptrtoint ptr %t1353 to i64
  store i64 %t1354, ptr %local.cmd.1069
  %t1355 = load i64, ptr %local.cmd.1069
  %t1356 = load i64, ptr %local.ll_path.1063
  %t1357 = inttoptr i64 %t1355 to ptr
  %t1358 = inttoptr i64 %t1356 to ptr
  %t1359 = call ptr @intrinsic_string_concat(ptr %t1357, ptr %t1358)
  %t1360 = ptrtoint ptr %t1359 to i64
  store i64 %t1360, ptr %local.cmd.1069
  %t1361 = load i64, ptr %local.cmd.1069
  %t1362 = inttoptr i64 %t1361 to ptr
  %t1363 = call i64 @intrinsic_process_run(ptr %t1362)
  store i64 %t1363, ptr %local.result.1070
  %t1364 = load i64, ptr %local.result.1070
  %t1365 = icmp ne i64 %t1364, 0
  %t1366 = zext i1 %t1365 to i64
  %t1367 = icmp ne i64 %t1366, 0
  br i1 %t1367, label %then98, label %else98
then98:
  %t1368 = call ptr @intrinsic_string_new(ptr @.str.main.73)
  %t1369 = ptrtoint ptr %t1368 to i64
  %t1370 = inttoptr i64 %t1369 to ptr
  call void @intrinsic_println(ptr %t1370)
  ret i64 1
  br label %then98_end
then98_end:
  br label %endif98
else98:
  br label %else98_end
else98_end:
  br label %endif98
endif98:
  %t1371 = phi i64 [ 0, %then98_end ], [ 0, %else98_end ]
  %t1372 = load i64, ptr %local.verbose
  %t1373 = icmp eq i64 %t1372, 1
  %t1374 = zext i1 %t1373 to i64
  %t1375 = icmp ne i64 %t1374, 0
  br i1 %t1375, label %then99, label %else99
then99:
  %t1376 = call ptr @intrinsic_string_new(ptr @.str.main.74)
  %t1377 = ptrtoint ptr %t1376 to i64
  %t1378 = load i64, ptr %local.obj_path.1068
  %t1379 = inttoptr i64 %t1377 to ptr
  %t1380 = inttoptr i64 %t1378 to ptr
  %t1381 = call ptr @intrinsic_string_concat(ptr %t1379, ptr %t1380)
  %t1382 = ptrtoint ptr %t1381 to i64
  %t1383 = inttoptr i64 %t1382 to ptr
  call void @intrinsic_println(ptr %t1383)
  br label %then99_end
then99_end:
  br label %endif99
else99:
  br label %else99_end
else99_end:
  br label %endif99
endif99:
  %t1384 = phi i64 [ 0, %then99_end ], [ 0, %else99_end ]
  ret i64 0
  br label %then96_end
then96_end:
  br label %endif96
else96:
  br label %else96_end
else96_end:
  br label %endif96
endif96:
  %t1385 = phi i64 [ 0, %then96_end ], [ 0, %else96_end ]
  %t1386 = load i64, ptr %local.emit_type
  %t1387 = icmp eq i64 %t1386, 2
  %t1388 = zext i1 %t1387 to i64
  %t1389 = icmp ne i64 %t1388, 0
  br i1 %t1389, label %then100, label %else100
then100:
  %t1390 = load i64, ptr %local.out_path.1064
  store i64 %t1390, ptr %local.exe_path.1071
  %t1391 = load i64, ptr %local.exe_path.1071
  %t1392 = icmp eq i64 %t1391, 0
  %t1393 = zext i1 %t1392 to i64
  %t1394 = icmp ne i64 %t1393, 0
  br i1 %t1394, label %then101, label %else101
then101:
  %t1395 = load i64, ptr %local.path
  %t1396 = call ptr @intrinsic_string_new(ptr @.str.main.75)
  %t1397 = ptrtoint ptr %t1396 to i64
  %t1398 = call i64 @"string_replace_ext"(i64 %t1395, i64 %t1397)
  store i64 %t1398, ptr %local.exe_path.1071
  br label %then101_end
then101_end:
  br label %endif101
else101:
  br label %else101_end
else101_end:
  br label %endif101
endif101:
  %t1399 = phi i64 [ 0, %then101_end ], [ 0, %else101_end ]
  %t1400 = load i64, ptr %local.path
  %t1401 = call ptr @intrinsic_string_new(ptr @.str.main.76)
  %t1402 = ptrtoint ptr %t1401 to i64
  %t1403 = call i64 @"string_replace_ext"(i64 %t1400, i64 %t1402)
  store i64 %t1403, ptr %local.obj_path.1072
  %t1404 = call ptr @intrinsic_string_new(ptr @.str.main.77)
  %t1405 = ptrtoint ptr %t1404 to i64
  %t1406 = load i64, ptr %local.obj_path.1072
  %t1407 = inttoptr i64 %t1405 to ptr
  %t1408 = inttoptr i64 %t1406 to ptr
  %t1409 = call ptr @intrinsic_string_concat(ptr %t1407, ptr %t1408)
  %t1410 = ptrtoint ptr %t1409 to i64
  store i64 %t1410, ptr %local.cmd.1073
  %t1411 = load i64, ptr %local.cmd.1073
  %t1412 = call ptr @intrinsic_string_new(ptr @.str.main.78)
  %t1413 = ptrtoint ptr %t1412 to i64
  %t1414 = inttoptr i64 %t1411 to ptr
  %t1415 = inttoptr i64 %t1413 to ptr
  %t1416 = call ptr @intrinsic_string_concat(ptr %t1414, ptr %t1415)
  %t1417 = ptrtoint ptr %t1416 to i64
  store i64 %t1417, ptr %local.cmd.1073
  %t1418 = load i64, ptr %local.cmd.1073
  %t1419 = load i64, ptr %local.ll_path.1063
  %t1420 = inttoptr i64 %t1418 to ptr
  %t1421 = inttoptr i64 %t1419 to ptr
  %t1422 = call ptr @intrinsic_string_concat(ptr %t1420, ptr %t1421)
  %t1423 = ptrtoint ptr %t1422 to i64
  store i64 %t1423, ptr %local.cmd.1073
  %t1424 = load i64, ptr %local.cmd.1073
  %t1425 = inttoptr i64 %t1424 to ptr
  %t1426 = call i64 @intrinsic_process_run(ptr %t1425)
  store i64 %t1426, ptr %local.result.1074
  %t1427 = load i64, ptr %local.result.1074
  %t1428 = icmp ne i64 %t1427, 0
  %t1429 = zext i1 %t1428 to i64
  %t1430 = icmp ne i64 %t1429, 0
  br i1 %t1430, label %then102, label %else102
then102:
  %t1431 = call ptr @intrinsic_string_new(ptr @.str.main.79)
  %t1432 = ptrtoint ptr %t1431 to i64
  %t1433 = inttoptr i64 %t1432 to ptr
  call void @intrinsic_println(ptr %t1433)
  ret i64 1
  br label %then102_end
then102_end:
  br label %endif102
else102:
  br label %else102_end
else102_end:
  br label %endif102
endif102:
  %t1434 = phi i64 [ 0, %then102_end ], [ 0, %else102_end ]
  %t1435 = call ptr @intrinsic_string_new(ptr @.str.main.80)
  %t1436 = ptrtoint ptr %t1435 to i64
  %t1437 = load i64, ptr %local.exe_path.1071
  %t1438 = inttoptr i64 %t1436 to ptr
  %t1439 = inttoptr i64 %t1437 to ptr
  %t1440 = call ptr @intrinsic_string_concat(ptr %t1438, ptr %t1439)
  %t1441 = ptrtoint ptr %t1440 to i64
  store i64 %t1441, ptr %local.cmd.1073
  %t1442 = load i64, ptr %local.cmd.1073
  %t1443 = call ptr @intrinsic_string_new(ptr @.str.main.81)
  %t1444 = ptrtoint ptr %t1443 to i64
  %t1445 = inttoptr i64 %t1442 to ptr
  %t1446 = inttoptr i64 %t1444 to ptr
  %t1447 = call ptr @intrinsic_string_concat(ptr %t1445, ptr %t1446)
  %t1448 = ptrtoint ptr %t1447 to i64
  store i64 %t1448, ptr %local.cmd.1073
  %t1449 = load i64, ptr %local.cmd.1073
  %t1450 = load i64, ptr %local.obj_path.1072
  %t1451 = inttoptr i64 %t1449 to ptr
  %t1452 = inttoptr i64 %t1450 to ptr
  %t1453 = call ptr @intrinsic_string_concat(ptr %t1451, ptr %t1452)
  %t1454 = ptrtoint ptr %t1453 to i64
  store i64 %t1454, ptr %local.cmd.1073
  %t1455 = load i64, ptr %local.cmd.1073
  %t1456 = call ptr @intrinsic_string_new(ptr @.str.main.82)
  %t1457 = ptrtoint ptr %t1456 to i64
  %t1458 = inttoptr i64 %t1455 to ptr
  %t1459 = inttoptr i64 %t1457 to ptr
  %t1460 = call ptr @intrinsic_string_concat(ptr %t1458, ptr %t1459)
  %t1461 = ptrtoint ptr %t1460 to i64
  store i64 %t1461, ptr %local.cmd.1073
  %t1462 = load i64, ptr %local.cmd.1073
  %t1463 = inttoptr i64 %t1462 to ptr
  %t1464 = call i64 @intrinsic_process_run(ptr %t1463)
  store i64 %t1464, ptr %local.result.1074
  %t1465 = load i64, ptr %local.result.1074
  %t1466 = icmp ne i64 %t1465, 0
  %t1467 = zext i1 %t1466 to i64
  %t1468 = icmp ne i64 %t1467, 0
  br i1 %t1468, label %then103, label %else103
then103:
  %t1469 = call ptr @intrinsic_string_new(ptr @.str.main.83)
  %t1470 = ptrtoint ptr %t1469 to i64
  %t1471 = inttoptr i64 %t1470 to ptr
  call void @intrinsic_println(ptr %t1471)
  ret i64 1
  br label %then103_end
then103_end:
  br label %endif103
else103:
  br label %else103_end
else103_end:
  br label %endif103
endif103:
  %t1472 = phi i64 [ 0, %then103_end ], [ 0, %else103_end ]
  %t1473 = load i64, ptr %local.verbose
  %t1474 = icmp eq i64 %t1473, 1
  %t1475 = zext i1 %t1474 to i64
  %t1476 = icmp ne i64 %t1475, 0
  br i1 %t1476, label %then104, label %else104
then104:
  %t1477 = call ptr @intrinsic_string_new(ptr @.str.main.84)
  %t1478 = ptrtoint ptr %t1477 to i64
  %t1479 = load i64, ptr %local.exe_path.1071
  %t1480 = inttoptr i64 %t1478 to ptr
  %t1481 = inttoptr i64 %t1479 to ptr
  %t1482 = call ptr @intrinsic_string_concat(ptr %t1480, ptr %t1481)
  %t1483 = ptrtoint ptr %t1482 to i64
  %t1484 = inttoptr i64 %t1483 to ptr
  call void @intrinsic_println(ptr %t1484)
  br label %then104_end
then104_end:
  br label %endif104
else104:
  br label %else104_end
else104_end:
  br label %endif104
endif104:
  %t1485 = phi i64 [ 0, %then104_end ], [ 0, %else104_end ]
  ret i64 0
  br label %then100_end
then100_end:
  br label %endif100
else100:
  br label %else100_end
else100_end:
  br label %endif100
endif100:
  %t1486 = phi i64 [ 0, %then100_end ], [ 0, %else100_end ]
  %t1487 = load i64, ptr %local.emit_type
  %t1488 = icmp eq i64 %t1487, 4
  %t1489 = zext i1 %t1488 to i64
  %t1490 = icmp ne i64 %t1489, 0
  br i1 %t1490, label %then105, label %else105
then105:
  %t1491 = load i64, ptr %local.out_path.1064
  store i64 %t1491, ptr %local.dylib_path.1075
  %t1492 = load i64, ptr %local.dylib_path.1075
  %t1493 = icmp eq i64 %t1492, 0
  %t1494 = zext i1 %t1493 to i64
  %t1495 = icmp ne i64 %t1494, 0
  br i1 %t1495, label %then106, label %else106
then106:
  %t1496 = load i64, ptr %local.path
  %t1497 = call ptr @intrinsic_string_new(ptr @.str.main.85)
  %t1498 = ptrtoint ptr %t1497 to i64
  %t1499 = call i64 @"string_replace_ext"(i64 %t1496, i64 %t1498)
  store i64 %t1499, ptr %local.dylib_path.1075
  br label %then106_end
then106_end:
  br label %endif106
else106:
  br label %else106_end
else106_end:
  br label %endif106
endif106:
  %t1500 = phi i64 [ 0, %then106_end ], [ 0, %else106_end ]
  %t1501 = load i64, ptr %local.path
  %t1502 = call ptr @intrinsic_string_new(ptr @.str.main.86)
  %t1503 = ptrtoint ptr %t1502 to i64
  %t1504 = call i64 @"string_replace_ext"(i64 %t1501, i64 %t1503)
  store i64 %t1504, ptr %local.obj_path.1076
  %t1505 = call ptr @intrinsic_string_new(ptr @.str.main.87)
  %t1506 = ptrtoint ptr %t1505 to i64
  %t1507 = load i64, ptr %local.obj_path.1076
  %t1508 = inttoptr i64 %t1506 to ptr
  %t1509 = inttoptr i64 %t1507 to ptr
  %t1510 = call ptr @intrinsic_string_concat(ptr %t1508, ptr %t1509)
  %t1511 = ptrtoint ptr %t1510 to i64
  store i64 %t1511, ptr %local.cmd.1077
  %t1512 = load i64, ptr %local.cmd.1077
  %t1513 = call ptr @intrinsic_string_new(ptr @.str.main.88)
  %t1514 = ptrtoint ptr %t1513 to i64
  %t1515 = inttoptr i64 %t1512 to ptr
  %t1516 = inttoptr i64 %t1514 to ptr
  %t1517 = call ptr @intrinsic_string_concat(ptr %t1515, ptr %t1516)
  %t1518 = ptrtoint ptr %t1517 to i64
  store i64 %t1518, ptr %local.cmd.1077
  %t1519 = load i64, ptr %local.cmd.1077
  %t1520 = load i64, ptr %local.ll_path.1063
  %t1521 = inttoptr i64 %t1519 to ptr
  %t1522 = inttoptr i64 %t1520 to ptr
  %t1523 = call ptr @intrinsic_string_concat(ptr %t1521, ptr %t1522)
  %t1524 = ptrtoint ptr %t1523 to i64
  store i64 %t1524, ptr %local.cmd.1077
  %t1525 = load i64, ptr %local.cmd.1077
  %t1526 = inttoptr i64 %t1525 to ptr
  %t1527 = call i64 @intrinsic_process_run(ptr %t1526)
  store i64 %t1527, ptr %local.result.1078
  %t1528 = load i64, ptr %local.result.1078
  %t1529 = icmp ne i64 %t1528, 0
  %t1530 = zext i1 %t1529 to i64
  %t1531 = icmp ne i64 %t1530, 0
  br i1 %t1531, label %then107, label %else107
then107:
  %t1532 = call ptr @intrinsic_string_new(ptr @.str.main.89)
  %t1533 = ptrtoint ptr %t1532 to i64
  %t1534 = inttoptr i64 %t1533 to ptr
  call void @intrinsic_println(ptr %t1534)
  ret i64 1
  br label %then107_end
then107_end:
  br label %endif107
else107:
  br label %else107_end
else107_end:
  br label %endif107
endif107:
  %t1535 = phi i64 [ 0, %then107_end ], [ 0, %else107_end ]
  %t1536 = call ptr @intrinsic_string_new(ptr @.str.main.90)
  %t1537 = ptrtoint ptr %t1536 to i64
  %t1538 = load i64, ptr %local.dylib_path.1075
  %t1539 = inttoptr i64 %t1537 to ptr
  %t1540 = inttoptr i64 %t1538 to ptr
  %t1541 = call ptr @intrinsic_string_concat(ptr %t1539, ptr %t1540)
  %t1542 = ptrtoint ptr %t1541 to i64
  store i64 %t1542, ptr %local.cmd.1077
  %t1543 = load i64, ptr %local.cmd.1077
  %t1544 = call ptr @intrinsic_string_new(ptr @.str.main.91)
  %t1545 = ptrtoint ptr %t1544 to i64
  %t1546 = inttoptr i64 %t1543 to ptr
  %t1547 = inttoptr i64 %t1545 to ptr
  %t1548 = call ptr @intrinsic_string_concat(ptr %t1546, ptr %t1547)
  %t1549 = ptrtoint ptr %t1548 to i64
  store i64 %t1549, ptr %local.cmd.1077
  %t1550 = load i64, ptr %local.cmd.1077
  %t1551 = load i64, ptr %local.obj_path.1076
  %t1552 = inttoptr i64 %t1550 to ptr
  %t1553 = inttoptr i64 %t1551 to ptr
  %t1554 = call ptr @intrinsic_string_concat(ptr %t1552, ptr %t1553)
  %t1555 = ptrtoint ptr %t1554 to i64
  store i64 %t1555, ptr %local.cmd.1077
  %t1556 = load i64, ptr %local.cmd.1077
  %t1557 = inttoptr i64 %t1556 to ptr
  %t1558 = call i64 @intrinsic_process_run(ptr %t1557)
  store i64 %t1558, ptr %local.result.1078
  %t1559 = load i64, ptr %local.result.1078
  %t1560 = icmp ne i64 %t1559, 0
  %t1561 = zext i1 %t1560 to i64
  %t1562 = icmp ne i64 %t1561, 0
  br i1 %t1562, label %then108, label %else108
then108:
  %t1563 = call ptr @intrinsic_string_new(ptr @.str.main.92)
  %t1564 = ptrtoint ptr %t1563 to i64
  %t1565 = inttoptr i64 %t1564 to ptr
  call void @intrinsic_println(ptr %t1565)
  ret i64 1
  br label %then108_end
then108_end:
  br label %endif108
else108:
  br label %else108_end
else108_end:
  br label %endif108
endif108:
  %t1566 = phi i64 [ 0, %then108_end ], [ 0, %else108_end ]
  %t1567 = load i64, ptr %local.verbose
  %t1568 = icmp eq i64 %t1567, 1
  %t1569 = zext i1 %t1568 to i64
  %t1570 = icmp ne i64 %t1569, 0
  br i1 %t1570, label %then109, label %else109
then109:
  %t1571 = call ptr @intrinsic_string_new(ptr @.str.main.93)
  %t1572 = ptrtoint ptr %t1571 to i64
  %t1573 = load i64, ptr %local.dylib_path.1075
  %t1574 = inttoptr i64 %t1572 to ptr
  %t1575 = inttoptr i64 %t1573 to ptr
  %t1576 = call ptr @intrinsic_string_concat(ptr %t1574, ptr %t1575)
  %t1577 = ptrtoint ptr %t1576 to i64
  %t1578 = inttoptr i64 %t1577 to ptr
  call void @intrinsic_println(ptr %t1578)
  br label %then109_end
then109_end:
  br label %endif109
else109:
  br label %else109_end
else109_end:
  br label %endif109
endif109:
  %t1579 = phi i64 [ 0, %then109_end ], [ 0, %else109_end ]
  ret i64 0
  br label %then105_end
then105_end:
  br label %endif105
else105:
  br label %else105_end
else105_end:
  br label %endif105
endif105:
  %t1580 = phi i64 [ 0, %then105_end ], [ 0, %else105_end ]
  ret i64 0
}

define i64 @"simplex_main"() {
entry:
  %local.args.1581 = alloca i64
  %local.argc.1582 = alloca i64
  %local.cli.1583 = alloca i64
  %local.cmd.1584 = alloca i64
  %local.files.1585 = alloca i64
  %local.file_count.1586 = alloca i64
  %local.i.1587 = alloca i64
  %local.path.1588 = alloca i64
  %local.deps.1589 = alloca i64
  %local.n.1590 = alloca i64
  %local.j.1591 = alloca i64
  %local.dep.1592 = alloca i64
  %local.verbose.1593 = alloca i64
  %local.emit.1594 = alloca i64
  %local.force.1595 = alloca i64
  %local.auto_deps.1596 = alloca i64
  %local.debug_info.1597 = alloca i64
  %local.i.1598 = alloca i64
  %local.path.1599 = alloca i64
  %local.out.1600 = alloca i64
  %local.result.1601 = alloca i64
  %local.path.1602 = alloca i64
  %local.result.1603 = alloca i64
  %local.exe_path.1604 = alloca i64
  %local.run_result.1605 = alloca i64
  %local.i.1606 = alloca i64
  %local.path.1607 = alloca i64
  %local.source.1608 = alloca i64
  %local.tokens.1609 = alloca i64
  %local.parse_result.1610 = alloca i64
  %local.parse_errors.1611 = alloca i64
  %local.i.1612 = alloca i64
  %local.path.1613 = alloca i64
  %local.result.1614 = alloca i64
  %t1615 = call ptr @intrinsic_get_args()
  %t1616 = ptrtoint ptr %t1615 to i64
  store i64 %t1616, ptr %local.args.1581
  %t1617 = load i64, ptr %local.args.1581
  %t1618 = inttoptr i64 %t1617 to ptr
  %t1619 = call i64 @intrinsic_vec_len(ptr %t1618)
  store i64 %t1619, ptr %local.argc.1582
  %t1620 = load i64, ptr %local.argc.1582
  %t1621 = icmp slt i64 %t1620, 2
  %t1622 = zext i1 %t1621 to i64
  %t1623 = icmp ne i64 %t1622, 0
  br i1 %t1623, label %then110, label %else110
then110:
  %t1624 = call i64 @"show_help"()
  ret i64 1
  br label %then110_end
then110_end:
  br label %endif110
else110:
  br label %else110_end
else110_end:
  br label %endif110
endif110:
  %t1625 = phi i64 [ 0, %then110_end ], [ 0, %else110_end ]
  %t1626 = load i64, ptr %local.args.1581
  %t1627 = call i64 @"parse_args"(i64 %t1626)
  store i64 %t1627, ptr %local.cli.1583
  %t1628 = load i64, ptr %local.cli.1583
  %t1629 = call i64 @"cli_command"(i64 %t1628)
  store i64 %t1629, ptr %local.cmd.1584
  %t1630 = load i64, ptr %local.cmd.1584
  %t1631 = icmp eq i64 %t1630, 3
  %t1632 = zext i1 %t1631 to i64
  %t1633 = icmp ne i64 %t1632, 0
  br i1 %t1633, label %then111, label %else111
then111:
  %t1634 = call i64 @"show_help"()
  ret i64 0
  br label %then111_end
then111_end:
  br label %endif111
else111:
  br label %else111_end
else111_end:
  br label %endif111
endif111:
  %t1635 = phi i64 [ 0, %then111_end ], [ 0, %else111_end ]
  %t1636 = load i64, ptr %local.cmd.1584
  %t1637 = icmp eq i64 %t1636, 4
  %t1638 = zext i1 %t1637 to i64
  %t1639 = icmp ne i64 %t1638, 0
  br i1 %t1639, label %then112, label %else112
then112:
  %t1640 = call i64 @"show_version"()
  ret i64 0
  br label %then112_end
then112_end:
  br label %endif112
else112:
  br label %else112_end
else112_end:
  br label %endif112
endif112:
  %t1641 = phi i64 [ 0, %then112_end ], [ 0, %else112_end ]
  %t1642 = load i64, ptr %local.cmd.1584
  %t1643 = icmp eq i64 %t1642, 5
  %t1644 = zext i1 %t1643 to i64
  %t1645 = icmp ne i64 %t1644, 0
  br i1 %t1645, label %then113, label %else113
then113:
  %t1646 = call i64 @"run_repl"()
  ret i64 %t1646
  br label %then113_end
then113_end:
  br label %endif113
else113:
  br label %else113_end
else113_end:
  br label %endif113
endif113:
  %t1647 = phi i64 [ 0, %then113_end ], [ 0, %else113_end ]
  %t1648 = load i64, ptr %local.cli.1583
  %t1649 = call i64 @"cli_input_files"(i64 %t1648)
  store i64 %t1649, ptr %local.files.1585
  %t1650 = load i64, ptr %local.files.1585
  %t1651 = inttoptr i64 %t1650 to ptr
  %t1652 = call i64 @intrinsic_vec_len(ptr %t1651)
  store i64 %t1652, ptr %local.file_count.1586
  %t1653 = load i64, ptr %local.file_count.1586
  %t1654 = icmp eq i64 %t1653, 0
  %t1655 = zext i1 %t1654 to i64
  %t1656 = icmp ne i64 %t1655, 0
  br i1 %t1656, label %then114, label %else114
then114:
  %t1657 = call ptr @intrinsic_string_new(ptr @.str.main.94)
  %t1658 = ptrtoint ptr %t1657 to i64
  %t1659 = inttoptr i64 %t1658 to ptr
  call void @intrinsic_println(ptr %t1659)
  ret i64 1
  br label %then114_end
then114_end:
  br label %endif114
else114:
  br label %else114_end
else114_end:
  br label %endif114
endif114:
  %t1660 = phi i64 [ 0, %then114_end ], [ 0, %else114_end ]
  %t1661 = load i64, ptr %local.cli.1583
  %t1662 = call i64 @"cli_show_deps"(i64 %t1661)
  %t1663 = icmp eq i64 %t1662, 1
  %t1664 = zext i1 %t1663 to i64
  %t1665 = icmp ne i64 %t1664, 0
  br i1 %t1665, label %then115, label %else115
then115:
  store i64 0, ptr %local.i.1587
  br label %loop116
loop116:
  %t1666 = load i64, ptr %local.i.1587
  %t1667 = load i64, ptr %local.file_count.1586
  %t1668 = icmp slt i64 %t1666, %t1667
  %t1669 = zext i1 %t1668 to i64
  %t1670 = icmp ne i64 %t1669, 0
  br i1 %t1670, label %body116, label %endloop116
body116:
  %t1671 = load i64, ptr %local.files.1585
  %t1672 = load i64, ptr %local.i.1587
  %t1673 = inttoptr i64 %t1671 to ptr
  %t1674 = call ptr @intrinsic_vec_get(ptr %t1673, i64 %t1672)
  %t1675 = ptrtoint ptr %t1674 to i64
  store i64 %t1675, ptr %local.path.1588
  %t1676 = load i64, ptr %local.path.1588
  %t1677 = call ptr @intrinsic_string_new(ptr @.str.main.95)
  %t1678 = ptrtoint ptr %t1677 to i64
  %t1679 = inttoptr i64 %t1676 to ptr
  %t1680 = inttoptr i64 %t1678 to ptr
  %t1681 = call ptr @intrinsic_string_concat(ptr %t1679, ptr %t1680)
  %t1682 = ptrtoint ptr %t1681 to i64
  %t1683 = inttoptr i64 %t1682 to ptr
  call void @intrinsic_println(ptr %t1683)
  %t1684 = load i64, ptr %local.path.1588
  %t1685 = call i64 @"extract_dependencies"(i64 %t1684)
  store i64 %t1685, ptr %local.deps.1589
  %t1686 = load i64, ptr %local.deps.1589
  %t1687 = inttoptr i64 %t1686 to ptr
  %t1688 = call i64 @intrinsic_vec_len(ptr %t1687)
  store i64 %t1688, ptr %local.n.1590
  %t1689 = load i64, ptr %local.n.1590
  %t1690 = icmp eq i64 %t1689, 0
  %t1691 = zext i1 %t1690 to i64
  %t1692 = icmp ne i64 %t1691, 0
  br i1 %t1692, label %then117, label %else117
then117:
  %t1693 = call ptr @intrinsic_string_new(ptr @.str.main.96)
  %t1694 = ptrtoint ptr %t1693 to i64
  %t1695 = inttoptr i64 %t1694 to ptr
  call void @intrinsic_println(ptr %t1695)
  br label %then117_end
then117_end:
  br label %endif117
else117:
  store i64 0, ptr %local.j.1591
  br label %loop118
loop118:
  %t1696 = load i64, ptr %local.j.1591
  %t1697 = load i64, ptr %local.n.1590
  %t1698 = icmp slt i64 %t1696, %t1697
  %t1699 = zext i1 %t1698 to i64
  %t1700 = icmp ne i64 %t1699, 0
  br i1 %t1700, label %body118, label %endloop118
body118:
  %t1701 = load i64, ptr %local.deps.1589
  %t1702 = load i64, ptr %local.j.1591
  %t1703 = inttoptr i64 %t1701 to ptr
  %t1704 = call ptr @intrinsic_vec_get(ptr %t1703, i64 %t1702)
  %t1705 = ptrtoint ptr %t1704 to i64
  store i64 %t1705, ptr %local.dep.1592
  %t1706 = call ptr @intrinsic_string_new(ptr @.str.main.97)
  %t1707 = ptrtoint ptr %t1706 to i64
  %t1708 = load i64, ptr %local.dep.1592
  %t1709 = inttoptr i64 %t1707 to ptr
  %t1710 = inttoptr i64 %t1708 to ptr
  %t1711 = call ptr @intrinsic_string_concat(ptr %t1709, ptr %t1710)
  %t1712 = ptrtoint ptr %t1711 to i64
  %t1713 = inttoptr i64 %t1712 to ptr
  call void @intrinsic_println(ptr %t1713)
  %t1714 = load i64, ptr %local.j.1591
  %t1715 = add i64 %t1714, 1
  store i64 %t1715, ptr %local.j.1591
  br label %loop118
endloop118:
  br label %else117_end
else117_end:
  br label %endif117
endif117:
  %t1716 = phi i64 [ 0, %then117_end ], [ 0, %else117_end ]
  %t1717 = load i64, ptr %local.i.1587
  %t1718 = add i64 %t1717, 1
  store i64 %t1718, ptr %local.i.1587
  br label %loop116
endloop116:
  ret i64 0
  br label %then115_end
then115_end:
  br label %endif115
else115:
  br label %else115_end
else115_end:
  br label %endif115
endif115:
  %t1719 = phi i64 [ 0, %then115_end ], [ 0, %else115_end ]
  %t1720 = load i64, ptr %local.cmd.1584
  %t1721 = icmp eq i64 %t1720, 0
  %t1722 = zext i1 %t1721 to i64
  %t1723 = icmp ne i64 %t1722, 0
  br i1 %t1723, label %then119, label %else119
then119:
  %t1724 = load i64, ptr %local.cli.1583
  %t1725 = call i64 @"cli_verbose"(i64 %t1724)
  store i64 %t1725, ptr %local.verbose.1593
  %t1726 = load i64, ptr %local.cli.1583
  %t1727 = call i64 @"cli_emit_type"(i64 %t1726)
  store i64 %t1727, ptr %local.emit.1594
  %t1728 = load i64, ptr %local.cli.1583
  %t1729 = call i64 @"cli_force"(i64 %t1728)
  store i64 %t1729, ptr %local.force.1595
  %t1730 = load i64, ptr %local.cli.1583
  %t1731 = call i64 @"cli_auto_deps"(i64 %t1730)
  store i64 %t1731, ptr %local.auto_deps.1596
  %t1732 = load i64, ptr %local.cli.1583
  %t1733 = call i64 @"cli_debug_info"(i64 %t1732)
  store i64 %t1733, ptr %local.debug_info.1597
  store i64 0, ptr %local.i.1598
  br label %loop120
loop120:
  %t1734 = load i64, ptr %local.i.1598
  %t1735 = load i64, ptr %local.file_count.1586
  %t1736 = icmp slt i64 %t1734, %t1735
  %t1737 = zext i1 %t1736 to i64
  %t1738 = icmp ne i64 %t1737, 0
  br i1 %t1738, label %body120, label %endloop120
body120:
  %t1739 = load i64, ptr %local.files.1585
  %t1740 = load i64, ptr %local.i.1598
  %t1741 = inttoptr i64 %t1739 to ptr
  %t1742 = call ptr @intrinsic_vec_get(ptr %t1741, i64 %t1740)
  %t1743 = ptrtoint ptr %t1742 to i64
  store i64 %t1743, ptr %local.path.1599
  store i64 0, ptr %local.out.1600
  %t1744 = load i64, ptr %local.file_count.1586
  %t1745 = icmp eq i64 %t1744, 1
  %t1746 = zext i1 %t1745 to i64
  %t1747 = icmp ne i64 %t1746, 0
  br i1 %t1747, label %then121, label %else121
then121:
  %t1748 = load i64, ptr %local.cli.1583
  %t1749 = call i64 @"cli_output_file"(i64 %t1748)
  store i64 %t1749, ptr %local.out.1600
  br label %then121_end
then121_end:
  br label %endif121
else121:
  br label %else121_end
else121_end:
  br label %endif121
endif121:
  %t1750 = phi i64 [ 0, %then121_end ], [ 0, %else121_end ]
  store i64 0, ptr %local.result.1601
  %t1751 = load i64, ptr %local.auto_deps.1596
  %t1752 = icmp eq i64 %t1751, 1
  %t1753 = zext i1 %t1752 to i64
  %t1754 = icmp ne i64 %t1753, 0
  br i1 %t1754, label %then122, label %else122
then122:
  %t1755 = load i64, ptr %local.path.1599
  %t1756 = load i64, ptr %local.out.1600
  %t1757 = load i64, ptr %local.verbose.1593
  %t1758 = load i64, ptr %local.emit.1594
  %t1759 = call i64 @"compile_with_dependencies"(i64 %t1755, i64 %t1756, i64 %t1757, i64 %t1758)
  store i64 %t1759, ptr %local.result.1601
  br label %then122_end
then122_end:
  br label %endif122
else122:
  %t1760 = load i64, ptr %local.path.1599
  %t1761 = load i64, ptr %local.out.1600
  %t1762 = load i64, ptr %local.verbose.1593
  %t1763 = load i64, ptr %local.emit.1594
  %t1764 = load i64, ptr %local.force.1595
  %t1765 = load i64, ptr %local.debug_info.1597
  %t1766 = call i64 @"compile_file_full"(i64 %t1760, i64 %t1761, i64 %t1762, i64 %t1763, i64 %t1764, i64 %t1765)
  store i64 %t1766, ptr %local.result.1601
  br label %else122_end
else122_end:
  br label %endif122
endif122:
  %t1767 = phi i64 [ 0, %then122_end ], [ 0, %else122_end ]
  %t1768 = load i64, ptr %local.result.1601
  %t1769 = icmp ne i64 %t1768, 0
  %t1770 = zext i1 %t1769 to i64
  %t1771 = icmp ne i64 %t1770, 0
  br i1 %t1771, label %then123, label %else123
then123:
  %t1772 = load i64, ptr %local.result.1601
  ret i64 %t1772
  br label %then123_end
then123_end:
  br label %endif123
else123:
  br label %else123_end
else123_end:
  br label %endif123
endif123:
  %t1773 = phi i64 [ 0, %then123_end ], [ 0, %else123_end ]
  %t1774 = load i64, ptr %local.i.1598
  %t1775 = add i64 %t1774, 1
  store i64 %t1775, ptr %local.i.1598
  br label %loop120
endloop120:
  %t1776 = call ptr @intrinsic_string_new(ptr @.str.main.98)
  %t1777 = ptrtoint ptr %t1776 to i64
  %t1778 = inttoptr i64 %t1777 to ptr
  call void @intrinsic_println(ptr %t1778)
  ret i64 0
  br label %then119_end
then119_end:
  br label %endif119
else119:
  br label %else119_end
else119_end:
  br label %endif119
endif119:
  %t1779 = phi i64 [ 0, %then119_end ], [ 0, %else119_end ]
  %t1780 = load i64, ptr %local.cmd.1584
  %t1781 = icmp eq i64 %t1780, 1
  %t1782 = zext i1 %t1781 to i64
  %t1783 = icmp ne i64 %t1782, 0
  br i1 %t1783, label %then124, label %else124
then124:
  %t1784 = load i64, ptr %local.file_count.1586
  %t1785 = icmp ne i64 %t1784, 1
  %t1786 = zext i1 %t1785 to i64
  %t1787 = icmp ne i64 %t1786, 0
  br i1 %t1787, label %then125, label %else125
then125:
  %t1788 = call ptr @intrinsic_string_new(ptr @.str.main.99)
  %t1789 = ptrtoint ptr %t1788 to i64
  %t1790 = inttoptr i64 %t1789 to ptr
  call void @intrinsic_println(ptr %t1790)
  ret i64 1
  br label %then125_end
then125_end:
  br label %endif125
else125:
  br label %else125_end
else125_end:
  br label %endif125
endif125:
  %t1791 = phi i64 [ 0, %then125_end ], [ 0, %else125_end ]
  %t1792 = load i64, ptr %local.files.1585
  %t1793 = inttoptr i64 %t1792 to ptr
  %t1794 = call ptr @intrinsic_vec_get(ptr %t1793, i64 0)
  %t1795 = ptrtoint ptr %t1794 to i64
  store i64 %t1795, ptr %local.path.1602
  %t1796 = load i64, ptr %local.path.1602
  %t1797 = call i64 @"compile_file"(i64 %t1796, i64 0, i64 0, i64 2)
  store i64 %t1797, ptr %local.result.1603
  %t1798 = load i64, ptr %local.result.1603
  %t1799 = icmp ne i64 %t1798, 0
  %t1800 = zext i1 %t1799 to i64
  %t1801 = icmp ne i64 %t1800, 0
  br i1 %t1801, label %then126, label %else126
then126:
  %t1802 = load i64, ptr %local.result.1603
  ret i64 %t1802
  br label %then126_end
then126_end:
  br label %endif126
else126:
  br label %else126_end
else126_end:
  br label %endif126
endif126:
  %t1803 = phi i64 [ 0, %then126_end ], [ 0, %else126_end ]
  %t1804 = load i64, ptr %local.path.1602
  %t1805 = call ptr @intrinsic_string_new(ptr @.str.main.100)
  %t1806 = ptrtoint ptr %t1805 to i64
  %t1807 = call i64 @"string_replace_ext"(i64 %t1804, i64 %t1806)
  store i64 %t1807, ptr %local.exe_path.1604
  %t1808 = load i64, ptr %local.exe_path.1604
  %t1809 = inttoptr i64 %t1808 to ptr
  %t1810 = call i64 @intrinsic_process_run(ptr %t1809)
  store i64 %t1810, ptr %local.run_result.1605
  %t1811 = load i64, ptr %local.run_result.1605
  ret i64 %t1811
  br label %then124_end
then124_end:
  br label %endif124
else124:
  br label %else124_end
else124_end:
  br label %endif124
endif124:
  %t1812 = phi i64 [ 0, %then124_end ], [ 0, %else124_end ]
  %t1813 = load i64, ptr %local.cmd.1584
  %t1814 = icmp eq i64 %t1813, 2
  %t1815 = zext i1 %t1814 to i64
  %t1816 = icmp ne i64 %t1815, 0
  br i1 %t1816, label %then127, label %else127
then127:
  store i64 0, ptr %local.i.1606
  br label %loop128
loop128:
  %t1817 = load i64, ptr %local.i.1606
  %t1818 = load i64, ptr %local.file_count.1586
  %t1819 = icmp slt i64 %t1817, %t1818
  %t1820 = zext i1 %t1819 to i64
  %t1821 = icmp ne i64 %t1820, 0
  br i1 %t1821, label %body128, label %endloop128
body128:
  %t1822 = load i64, ptr %local.files.1585
  %t1823 = load i64, ptr %local.i.1606
  %t1824 = inttoptr i64 %t1822 to ptr
  %t1825 = call ptr @intrinsic_vec_get(ptr %t1824, i64 %t1823)
  %t1826 = ptrtoint ptr %t1825 to i64
  store i64 %t1826, ptr %local.path.1607
  %t1827 = load i64, ptr %local.path.1607
  %t1828 = inttoptr i64 %t1827 to ptr
  %t1829 = call ptr @intrinsic_read_file(ptr %t1828)
  %t1830 = ptrtoint ptr %t1829 to i64
  store i64 %t1830, ptr %local.source.1608
  %t1831 = load i64, ptr %local.source.1608
  %t1832 = icmp eq i64 %t1831, 0
  %t1833 = zext i1 %t1832 to i64
  %t1834 = icmp ne i64 %t1833, 0
  br i1 %t1834, label %then129, label %else129
then129:
  %t1835 = call i64 @"E_CANNOT_READ_FILE"()
  %t1836 = call ptr @intrinsic_string_new(ptr @.str.main.101)
  %t1837 = ptrtoint ptr %t1836 to i64
  %t1838 = load i64, ptr %local.path.1607
  %t1839 = inttoptr i64 %t1837 to ptr
  %t1840 = inttoptr i64 %t1838 to ptr
  %t1841 = call ptr @intrinsic_string_concat(ptr %t1839, ptr %t1840)
  %t1842 = ptrtoint ptr %t1841 to i64
  %t1843 = call i64 @"report_simple_error"(i64 %t1835, i64 %t1842)
  ret i64 1
  br label %then129_end
then129_end:
  br label %endif129
else129:
  br label %else129_end
else129_end:
  br label %endif129
endif129:
  %t1844 = phi i64 [ 0, %then129_end ], [ 0, %else129_end ]
  %t1845 = load i64, ptr %local.source.1608
  %t1846 = call i64 @"tokenize"(i64 %t1845)
  store i64 %t1846, ptr %local.tokens.1609
  %t1847 = load i64, ptr %local.tokens.1609
  %t1848 = load i64, ptr %local.source.1608
  %t1849 = load i64, ptr %local.path.1607
  %t1850 = call i64 @"parse_program_with_source"(i64 %t1847, i64 %t1848, i64 %t1849)
  store i64 %t1850, ptr %local.parse_result.1610
  %t1851 = load i64, ptr %local.parse_result.1610
  %t1852 = call i64 @"parse_result_errors"(i64 %t1851)
  store i64 %t1852, ptr %local.parse_errors.1611
  %t1853 = load i64, ptr %local.parse_errors.1611
  %t1854 = icmp sgt i64 %t1853, 0
  %t1855 = zext i1 %t1854 to i64
  %t1856 = icmp ne i64 %t1855, 0
  br i1 %t1856, label %then130, label %else130
then130:
  %t1857 = load i64, ptr %local.parse_errors.1611
  %t1858 = call i64 @"print_error_summary"(i64 %t1857, i64 0)
  ret i64 1
  br label %then130_end
then130_end:
  br label %endif130
else130:
  br label %else130_end
else130_end:
  br label %endif130
endif130:
  %t1859 = phi i64 [ 0, %then130_end ], [ 0, %else130_end ]
  %t1860 = load i64, ptr %local.path.1607
  %t1861 = call ptr @intrinsic_string_new(ptr @.str.main.102)
  %t1862 = ptrtoint ptr %t1861 to i64
  %t1863 = inttoptr i64 %t1860 to ptr
  %t1864 = inttoptr i64 %t1862 to ptr
  %t1865 = call ptr @intrinsic_string_concat(ptr %t1863, ptr %t1864)
  %t1866 = ptrtoint ptr %t1865 to i64
  %t1867 = inttoptr i64 %t1866 to ptr
  call void @intrinsic_println(ptr %t1867)
  %t1868 = load i64, ptr %local.i.1606
  %t1869 = add i64 %t1868, 1
  store i64 %t1869, ptr %local.i.1606
  br label %loop128
endloop128:
  ret i64 0
  br label %then127_end
then127_end:
  br label %endif127
else127:
  br label %else127_end
else127_end:
  br label %endif127
endif127:
  %t1870 = phi i64 [ 0, %then127_end ], [ 0, %else127_end ]
  %t1871 = load i64, ptr %local.cmd.1584
  %t1872 = icmp eq i64 %t1871, 6
  %t1873 = zext i1 %t1872 to i64
  %t1874 = icmp ne i64 %t1873, 0
  br i1 %t1874, label %then131, label %else131
then131:
  %t1875 = load i64, ptr %local.file_count.1586
  %t1876 = icmp eq i64 %t1875, 0
  %t1877 = zext i1 %t1876 to i64
  %t1878 = icmp ne i64 %t1877, 0
  br i1 %t1878, label %then132, label %else132
then132:
  %t1879 = call ptr @intrinsic_string_new(ptr @.str.main.103)
  %t1880 = ptrtoint ptr %t1879 to i64
  %t1881 = inttoptr i64 %t1880 to ptr
  call void @intrinsic_println(ptr %t1881)
  ret i64 1
  br label %then132_end
then132_end:
  br label %endif132
else132:
  br label %else132_end
else132_end:
  br label %endif132
endif132:
  %t1882 = phi i64 [ 0, %then132_end ], [ 0, %else132_end ]
  store i64 0, ptr %local.i.1612
  br label %loop133
loop133:
  %t1883 = load i64, ptr %local.i.1612
  %t1884 = load i64, ptr %local.file_count.1586
  %t1885 = icmp slt i64 %t1883, %t1884
  %t1886 = zext i1 %t1885 to i64
  %t1887 = icmp ne i64 %t1886, 0
  br i1 %t1887, label %body133, label %endloop133
body133:
  %t1888 = load i64, ptr %local.files.1585
  %t1889 = load i64, ptr %local.i.1612
  %t1890 = inttoptr i64 %t1888 to ptr
  %t1891 = call ptr @intrinsic_vec_get(ptr %t1890, i64 %t1889)
  %t1892 = ptrtoint ptr %t1891 to i64
  store i64 %t1892, ptr %local.path.1613
  %t1893 = load i64, ptr %local.path.1613
  %t1894 = call i64 @"format_file"(i64 %t1893, i64 1)
  store i64 %t1894, ptr %local.result.1614
  %t1895 = load i64, ptr %local.result.1614
  %t1896 = icmp ne i64 %t1895, 0
  %t1897 = zext i1 %t1896 to i64
  %t1898 = icmp ne i64 %t1897, 0
  br i1 %t1898, label %then134, label %else134
then134:
  %t1899 = load i64, ptr %local.result.1614
  ret i64 %t1899
  br label %then134_end
then134_end:
  br label %endif134
else134:
  br label %else134_end
else134_end:
  br label %endif134
endif134:
  %t1900 = phi i64 [ 0, %then134_end ], [ 0, %else134_end ]
  %t1901 = load i64, ptr %local.i.1612
  %t1902 = add i64 %t1901, 1
  store i64 %t1902, ptr %local.i.1612
  br label %loop133
endloop133:
  ret i64 0
  br label %then131_end
then131_end:
  br label %endif131
else131:
  br label %else131_end
else131_end:
  br label %endif131
endif131:
  %t1903 = phi i64 [ 0, %then131_end ], [ 0, %else131_end ]
  ret i64 0
}

define i64 @"string_replace_ext"(i64 %path, i64 %new_ext) {
entry:
  %local.len.1904 = alloca i64
  %local.i.1905 = alloca i64
  %local.c.1906 = alloca i64
  %local.base.1907 = alloca i64
  %local.path = alloca i64
  store i64 %path, ptr %local.path
  %local.new_ext = alloca i64
  store i64 %new_ext, ptr %local.new_ext
  %t1908 = load i64, ptr %local.path
  %t1909 = inttoptr i64 %t1908 to ptr
  %t1910 = call i64 @intrinsic_string_len(ptr %t1909)
  store i64 %t1910, ptr %local.len.1904
  %t1911 = load i64, ptr %local.len.1904
  %t1912 = sub i64 %t1911, 1
  store i64 %t1912, ptr %local.i.1905
  br label %loop135
loop135:
  %t1913 = load i64, ptr %local.i.1905
  %t1914 = icmp sge i64 %t1913, 0
  %t1915 = zext i1 %t1914 to i64
  %t1916 = icmp ne i64 %t1915, 0
  br i1 %t1916, label %body135, label %endloop135
body135:
  %t1917 = load i64, ptr %local.path
  %t1918 = load i64, ptr %local.i.1905
  %t1919 = inttoptr i64 %t1917 to ptr
  %t1920 = call i64 @intrinsic_string_char_at(ptr %t1919, i64 %t1918)
  store i64 %t1920, ptr %local.c.1906
  %t1921 = load i64, ptr %local.c.1906
  %t1922 = icmp eq i64 %t1921, 46
  %t1923 = zext i1 %t1922 to i64
  %t1924 = icmp ne i64 %t1923, 0
  br i1 %t1924, label %then136, label %else136
then136:
  %t1925 = load i64, ptr %local.path
  %t1926 = load i64, ptr %local.i.1905
  %t1927 = inttoptr i64 %t1925 to ptr
  %t1928 = call ptr @intrinsic_string_slice(ptr %t1927, i64 0, i64 %t1926)
  %t1929 = ptrtoint ptr %t1928 to i64
  store i64 %t1929, ptr %local.base.1907
  %t1930 = load i64, ptr %local.base.1907
  %t1931 = load i64, ptr %local.new_ext
  %t1932 = inttoptr i64 %t1930 to ptr
  %t1933 = inttoptr i64 %t1931 to ptr
  %t1934 = call ptr @intrinsic_string_concat(ptr %t1932, ptr %t1933)
  %t1935 = ptrtoint ptr %t1934 to i64
  ret i64 %t1935
  br label %then136_end
then136_end:
  br label %endif136
else136:
  br label %else136_end
else136_end:
  br label %endif136
endif136:
  %t1936 = phi i64 [ 0, %then136_end ], [ 0, %else136_end ]
  %t1937 = load i64, ptr %local.i.1905
  %t1938 = sub i64 %t1937, 1
  store i64 %t1938, ptr %local.i.1905
  br label %loop135
endloop135:
  %t1939 = load i64, ptr %local.path
  %t1940 = load i64, ptr %local.new_ext
  %t1941 = inttoptr i64 %t1939 to ptr
  %t1942 = inttoptr i64 %t1940 to ptr
  %t1943 = call ptr @intrinsic_string_concat(ptr %t1941, ptr %t1942)
  %t1944 = ptrtoint ptr %t1943 to i64
  ret i64 %t1944
}

define i64 @"get_module_name"(i64 %path) {
entry:
  %local.len.1945 = alloca i64
  %local.start.1946 = alloca i64
  %local.i.1947 = alloca i64
  %local.c.1948 = alloca i64
  %local.end.1949 = alloca i64
  %local.c.1950 = alloca i64
  %local.path = alloca i64
  store i64 %path, ptr %local.path
  %t1951 = load i64, ptr %local.path
  %t1952 = inttoptr i64 %t1951 to ptr
  %t1953 = call i64 @intrinsic_string_len(ptr %t1952)
  store i64 %t1953, ptr %local.len.1945
  store i64 0, ptr %local.start.1946
  store i64 0, ptr %local.i.1947
  br label %loop137
loop137:
  %t1954 = load i64, ptr %local.i.1947
  %t1955 = load i64, ptr %local.len.1945
  %t1956 = icmp slt i64 %t1954, %t1955
  %t1957 = zext i1 %t1956 to i64
  %t1958 = icmp ne i64 %t1957, 0
  br i1 %t1958, label %body137, label %endloop137
body137:
  %t1959 = load i64, ptr %local.path
  %t1960 = load i64, ptr %local.i.1947
  %t1961 = inttoptr i64 %t1959 to ptr
  %t1962 = call i64 @intrinsic_string_char_at(ptr %t1961, i64 %t1960)
  store i64 %t1962, ptr %local.c.1948
  %t1963 = load i64, ptr %local.c.1948
  %t1964 = icmp eq i64 %t1963, 47
  %t1965 = zext i1 %t1964 to i64
  %t1966 = icmp ne i64 %t1965, 0
  br i1 %t1966, label %then138, label %else138
then138:
  %t1967 = load i64, ptr %local.i.1947
  %t1968 = add i64 %t1967, 1
  store i64 %t1968, ptr %local.start.1946
  br label %then138_end
then138_end:
  br label %endif138
else138:
  br label %else138_end
else138_end:
  br label %endif138
endif138:
  %t1969 = phi i64 [ 0, %then138_end ], [ 0, %else138_end ]
  %t1970 = load i64, ptr %local.i.1947
  %t1971 = add i64 %t1970, 1
  store i64 %t1971, ptr %local.i.1947
  br label %loop137
endloop137:
  %t1972 = load i64, ptr %local.len.1945
  store i64 %t1972, ptr %local.end.1949
  %t1973 = load i64, ptr %local.len.1945
  %t1974 = sub i64 %t1973, 1
  store i64 %t1974, ptr %local.i.1947
  br label %loop139
loop139:
  %t1975 = load i64, ptr %local.i.1947
  %t1976 = load i64, ptr %local.start.1946
  %t1977 = icmp sge i64 %t1975, %t1976
  %t1978 = zext i1 %t1977 to i64
  %t1979 = icmp ne i64 %t1978, 0
  br i1 %t1979, label %body139, label %endloop139
body139:
  %t1980 = load i64, ptr %local.path
  %t1981 = load i64, ptr %local.i.1947
  %t1982 = inttoptr i64 %t1980 to ptr
  %t1983 = call i64 @intrinsic_string_char_at(ptr %t1982, i64 %t1981)
  store i64 %t1983, ptr %local.c.1950
  %t1984 = load i64, ptr %local.c.1950
  %t1985 = icmp eq i64 %t1984, 46
  %t1986 = zext i1 %t1985 to i64
  %t1987 = icmp ne i64 %t1986, 0
  br i1 %t1987, label %then140, label %else140
then140:
  %t1988 = load i64, ptr %local.i.1947
  store i64 %t1988, ptr %local.end.1949
  store i64 0, ptr %local.i.1947
  br label %then140_end
then140_end:
  br label %endif140
else140:
  br label %else140_end
else140_end:
  br label %endif140
endif140:
  %t1989 = phi i64 [ 0, %then140_end ], [ 0, %else140_end ]
  %t1990 = load i64, ptr %local.i.1947
  %t1991 = sub i64 %t1990, 1
  store i64 %t1991, ptr %local.i.1947
  br label %loop139
endloop139:
  %t1992 = load i64, ptr %local.path
  %t1993 = load i64, ptr %local.start.1946
  %t1994 = load i64, ptr %local.end.1949
  %t1995 = inttoptr i64 %t1992 to ptr
  %t1996 = call ptr @intrinsic_string_slice(ptr %t1995, i64 %t1993, i64 %t1994)
  %t1997 = ptrtoint ptr %t1996 to i64
  ret i64 %t1997
}

define i64 @"repl_state_new"() {
entry:
  %local.state.1998 = alloca i64
  %t1999 = call ptr @malloc(i64 16)
  %t2000 = ptrtoint ptr %t1999 to i64
  store i64 %t2000, ptr %local.state.1998
  %t2001 = load i64, ptr %local.state.1998
  %t2002 = call ptr @intrinsic_vec_new()
  %t2003 = ptrtoint ptr %t2002 to i64
  %t2004 = inttoptr i64 %t2001 to ptr
  %t2005 = inttoptr i64 %t2003 to ptr
  %t2006 = call ptr @store_ptr(ptr %t2004, i64 0, ptr %t2005)
  %t2007 = ptrtoint ptr %t2006 to i64
  %t2008 = load i64, ptr %local.state.1998
  %t2009 = inttoptr i64 %t2008 to ptr
  %t2010 = call ptr @store_i64(ptr %t2009, i64 1, i64 0)
  %t2011 = ptrtoint ptr %t2010 to i64
  %t2012 = load i64, ptr %local.state.1998
  ret i64 %t2012
}

define i64 @"repl_state_vars"(i64 %state) {
entry:
  %local.state = alloca i64
  store i64 %state, ptr %local.state
  %t2013 = load i64, ptr %local.state
  %t2014 = inttoptr i64 %t2013 to ptr
  %t2015 = call ptr @load_ptr(ptr %t2014, i64 0)
  %t2016 = ptrtoint ptr %t2015 to i64
  ret i64 %t2016
}

define i64 @"run_repl"() {
entry:
  %local.state.2017 = alloca i64
  %local.line_num.2018 = alloca i64
  %local.line.2019 = alloca i64
  %local.trimmed.2020 = alloca i64
  %local.vars.2021 = alloca i64
  %local.n.2022 = alloca i64
  %local.i.2023 = alloca i64
  %local.pair.2024 = alloca i64
  %local.name.2025 = alloca i64
  %local.value.2026 = alloca i64
  %local.result.2027 = alloca i64
  %t2028 = call ptr @intrinsic_string_new(ptr @.str.main.104)
  %t2029 = ptrtoint ptr %t2028 to i64
  %t2030 = inttoptr i64 %t2029 to ptr
  call void @intrinsic_println(ptr %t2030)
  %t2031 = call ptr @intrinsic_string_new(ptr @.str.main.105)
  %t2032 = ptrtoint ptr %t2031 to i64
  %t2033 = inttoptr i64 %t2032 to ptr
  call void @intrinsic_println(ptr %t2033)
  %t2034 = call ptr @intrinsic_string_new(ptr @.str.main.106)
  %t2035 = ptrtoint ptr %t2034 to i64
  %t2036 = inttoptr i64 %t2035 to ptr
  call void @intrinsic_println(ptr %t2036)
  %t2037 = call i64 @"repl_state_new"()
  store i64 %t2037, ptr %local.state.2017
  store i64 1, ptr %local.line_num.2018
  br label %loop141
loop141:
  %t2038 = icmp ne i64 1, 0
  br i1 %t2038, label %body141, label %endloop141
body141:
  %t2039 = call ptr @intrinsic_string_new(ptr @.str.main.107)
  %t2040 = ptrtoint ptr %t2039 to i64
  %t2041 = load i64, ptr %local.line_num.2018
  %t2042 = call ptr @intrinsic_int_to_string(i64 %t2041)
  %t2043 = ptrtoint ptr %t2042 to i64
  %t2044 = call ptr @intrinsic_string_new(ptr @.str.main.108)
  %t2045 = ptrtoint ptr %t2044 to i64
  %t2046 = inttoptr i64 %t2043 to ptr
  %t2047 = inttoptr i64 %t2045 to ptr
  %t2048 = call ptr @intrinsic_string_concat(ptr %t2046, ptr %t2047)
  %t2049 = ptrtoint ptr %t2048 to i64
  %t2050 = inttoptr i64 %t2040 to ptr
  %t2051 = inttoptr i64 %t2049 to ptr
  %t2052 = call ptr @intrinsic_string_concat(ptr %t2050, ptr %t2051)
  %t2053 = ptrtoint ptr %t2052 to i64
  %t2054 = inttoptr i64 %t2053 to ptr
  call void @intrinsic_print(ptr %t2054)
  %t2055 = call ptr @intrinsic_read_line()
  %t2056 = ptrtoint ptr %t2055 to i64
  store i64 %t2056, ptr %local.line.2019
  %t2057 = load i64, ptr %local.line.2019
  %t2058 = icmp eq i64 %t2057, 0
  %t2059 = zext i1 %t2058 to i64
  %t2060 = icmp ne i64 %t2059, 0
  br i1 %t2060, label %then142, label %else142
then142:
  %t2061 = call ptr @intrinsic_string_new(ptr @.str.main.109)
  %t2062 = ptrtoint ptr %t2061 to i64
  %t2063 = inttoptr i64 %t2062 to ptr
  call void @intrinsic_println(ptr %t2063)
  %t2064 = call ptr @intrinsic_string_new(ptr @.str.main.110)
  %t2065 = ptrtoint ptr %t2064 to i64
  %t2066 = inttoptr i64 %t2065 to ptr
  call void @intrinsic_println(ptr %t2066)
  ret i64 0
  br label %then142_end
then142_end:
  br label %endif142
else142:
  br label %else142_end
else142_end:
  br label %endif142
endif142:
  %t2067 = phi i64 [ 0, %then142_end ], [ 0, %else142_end ]
  %t2068 = load i64, ptr %local.line.2019
  %t2069 = inttoptr i64 %t2068 to ptr
  %t2070 = call ptr @intrinsic_string_trim(ptr %t2069)
  %t2071 = ptrtoint ptr %t2070 to i64
  store i64 %t2071, ptr %local.trimmed.2020
  %t2072 = load i64, ptr %local.trimmed.2020
  %t2073 = call ptr @intrinsic_string_new(ptr @.str.main.111)
  %t2074 = ptrtoint ptr %t2073 to i64
  %t2075 = inttoptr i64 %t2072 to ptr
  %t2076 = inttoptr i64 %t2074 to ptr
  %t2077 = call i1 @intrinsic_string_eq(ptr %t2075, ptr %t2076)
  %t2078 = zext i1 %t2077 to i64
  %t2079 = icmp ne i64 %t2078, 0
  br i1 %t2079, label %then143, label %else143
then143:
  %t2080 = call ptr @intrinsic_string_new(ptr @.str.main.112)
  %t2081 = ptrtoint ptr %t2080 to i64
  %t2082 = inttoptr i64 %t2081 to ptr
  call void @intrinsic_println(ptr %t2082)
  ret i64 0
  br label %then143_end
then143_end:
  br label %endif143
else143:
  br label %else143_end
else143_end:
  br label %endif143
endif143:
  %t2083 = phi i64 [ 0, %then143_end ], [ 0, %else143_end ]
  %t2084 = load i64, ptr %local.trimmed.2020
  %t2085 = call ptr @intrinsic_string_new(ptr @.str.main.113)
  %t2086 = ptrtoint ptr %t2085 to i64
  %t2087 = inttoptr i64 %t2084 to ptr
  %t2088 = inttoptr i64 %t2086 to ptr
  %t2089 = call i1 @intrinsic_string_eq(ptr %t2087, ptr %t2088)
  %t2090 = zext i1 %t2089 to i64
  %t2091 = icmp ne i64 %t2090, 0
  br i1 %t2091, label %then144, label %else144
then144:
  %t2092 = call ptr @intrinsic_string_new(ptr @.str.main.114)
  %t2093 = ptrtoint ptr %t2092 to i64
  %t2094 = inttoptr i64 %t2093 to ptr
  call void @intrinsic_println(ptr %t2094)
  ret i64 0
  br label %then144_end
then144_end:
  br label %endif144
else144:
  br label %else144_end
else144_end:
  br label %endif144
endif144:
  %t2095 = phi i64 [ 0, %then144_end ], [ 0, %else144_end ]
  %t2096 = load i64, ptr %local.trimmed.2020
  %t2097 = inttoptr i64 %t2096 to ptr
  %t2098 = call i64 @intrinsic_string_len(ptr %t2097)
  %t2099 = icmp eq i64 %t2098, 0
  %t2100 = zext i1 %t2099 to i64
  %t2101 = icmp ne i64 %t2100, 0
  br i1 %t2101, label %then145, label %else145
then145:
  %t2102 = load i64, ptr %local.line_num.2018
  %t2103 = add i64 %t2102, 1
  store i64 %t2103, ptr %local.line_num.2018
  br label %then145_end
then145_end:
  br label %endif145
else145:
  %t2104 = load i64, ptr %local.trimmed.2020
  %t2105 = call ptr @intrinsic_string_new(ptr @.str.main.115)
  %t2106 = ptrtoint ptr %t2105 to i64
  %t2107 = inttoptr i64 %t2104 to ptr
  %t2108 = inttoptr i64 %t2106 to ptr
  %t2109 = call i1 @intrinsic_string_eq(ptr %t2107, ptr %t2108)
  %t2110 = zext i1 %t2109 to i64
  %t2111 = icmp ne i64 %t2110, 0
  br i1 %t2111, label %then146, label %else146
then146:
  %t2112 = call ptr @intrinsic_string_new(ptr @.str.main.116)
  %t2113 = ptrtoint ptr %t2112 to i64
  %t2114 = inttoptr i64 %t2113 to ptr
  call void @intrinsic_println(ptr %t2114)
  %t2115 = call ptr @intrinsic_string_new(ptr @.str.main.117)
  %t2116 = ptrtoint ptr %t2115 to i64
  %t2117 = inttoptr i64 %t2116 to ptr
  call void @intrinsic_println(ptr %t2117)
  %t2118 = call ptr @intrinsic_string_new(ptr @.str.main.118)
  %t2119 = ptrtoint ptr %t2118 to i64
  %t2120 = inttoptr i64 %t2119 to ptr
  call void @intrinsic_println(ptr %t2120)
  %t2121 = call ptr @intrinsic_string_new(ptr @.str.main.119)
  %t2122 = ptrtoint ptr %t2121 to i64
  %t2123 = inttoptr i64 %t2122 to ptr
  call void @intrinsic_println(ptr %t2123)
  %t2124 = call ptr @intrinsic_string_new(ptr @.str.main.120)
  %t2125 = ptrtoint ptr %t2124 to i64
  %t2126 = inttoptr i64 %t2125 to ptr
  call void @intrinsic_println(ptr %t2126)
  %t2127 = load i64, ptr %local.line_num.2018
  %t2128 = add i64 %t2127, 1
  store i64 %t2128, ptr %local.line_num.2018
  br label %then146_end
then146_end:
  br label %endif146
else146:
  %t2129 = load i64, ptr %local.trimmed.2020
  %t2130 = call ptr @intrinsic_string_new(ptr @.str.main.121)
  %t2131 = ptrtoint ptr %t2130 to i64
  %t2132 = inttoptr i64 %t2129 to ptr
  %t2133 = inttoptr i64 %t2131 to ptr
  %t2134 = call i1 @intrinsic_string_eq(ptr %t2132, ptr %t2133)
  %t2135 = zext i1 %t2134 to i64
  %t2136 = icmp ne i64 %t2135, 0
  br i1 %t2136, label %then147, label %else147
then147:
  %t2137 = load i64, ptr %local.state.2017
  %t2138 = call i64 @"repl_state_vars"(i64 %t2137)
  store i64 %t2138, ptr %local.vars.2021
  %t2139 = load i64, ptr %local.vars.2021
  %t2140 = inttoptr i64 %t2139 to ptr
  %t2141 = call i64 @intrinsic_vec_len(ptr %t2140)
  store i64 %t2141, ptr %local.n.2022
  %t2142 = load i64, ptr %local.n.2022
  %t2143 = icmp eq i64 %t2142, 0
  %t2144 = zext i1 %t2143 to i64
  %t2145 = icmp ne i64 %t2144, 0
  br i1 %t2145, label %then148, label %else148
then148:
  %t2146 = call ptr @intrinsic_string_new(ptr @.str.main.122)
  %t2147 = ptrtoint ptr %t2146 to i64
  %t2148 = inttoptr i64 %t2147 to ptr
  call void @intrinsic_println(ptr %t2148)
  br label %then148_end
then148_end:
  br label %endif148
else148:
  store i64 0, ptr %local.i.2023
  br label %loop149
loop149:
  %t2149 = load i64, ptr %local.i.2023
  %t2150 = load i64, ptr %local.n.2022
  %t2151 = icmp slt i64 %t2149, %t2150
  %t2152 = zext i1 %t2151 to i64
  %t2153 = icmp ne i64 %t2152, 0
  br i1 %t2153, label %body149, label %endloop149
body149:
  %t2154 = load i64, ptr %local.vars.2021
  %t2155 = load i64, ptr %local.i.2023
  %t2156 = inttoptr i64 %t2154 to ptr
  %t2157 = call ptr @intrinsic_vec_get(ptr %t2156, i64 %t2155)
  %t2158 = ptrtoint ptr %t2157 to i64
  store i64 %t2158, ptr %local.pair.2024
  %t2159 = load i64, ptr %local.pair.2024
  %t2160 = inttoptr i64 %t2159 to ptr
  %t2161 = call ptr @load_ptr(ptr %t2160, i64 0)
  %t2162 = ptrtoint ptr %t2161 to i64
  store i64 %t2162, ptr %local.name.2025
  %t2163 = load i64, ptr %local.pair.2024
  %t2164 = inttoptr i64 %t2163 to ptr
  %t2165 = call i64 @load_i64(ptr %t2164, i64 1)
  store i64 %t2165, ptr %local.value.2026
  %t2166 = load i64, ptr %local.name.2025
  %t2167 = call ptr @intrinsic_string_new(ptr @.str.main.123)
  %t2168 = ptrtoint ptr %t2167 to i64
  %t2169 = load i64, ptr %local.value.2026
  %t2170 = call ptr @intrinsic_int_to_string(i64 %t2169)
  %t2171 = ptrtoint ptr %t2170 to i64
  %t2172 = inttoptr i64 %t2168 to ptr
  %t2173 = inttoptr i64 %t2171 to ptr
  %t2174 = call ptr @intrinsic_string_concat(ptr %t2172, ptr %t2173)
  %t2175 = ptrtoint ptr %t2174 to i64
  %t2176 = inttoptr i64 %t2166 to ptr
  %t2177 = inttoptr i64 %t2175 to ptr
  %t2178 = call ptr @intrinsic_string_concat(ptr %t2176, ptr %t2177)
  %t2179 = ptrtoint ptr %t2178 to i64
  %t2180 = inttoptr i64 %t2179 to ptr
  call void @intrinsic_println(ptr %t2180)
  %t2181 = load i64, ptr %local.i.2023
  %t2182 = add i64 %t2181, 1
  store i64 %t2182, ptr %local.i.2023
  br label %loop149
endloop149:
  br label %else148_end
else148_end:
  br label %endif148
endif148:
  %t2183 = phi i64 [ 0, %then148_end ], [ 0, %else148_end ]
  %t2184 = load i64, ptr %local.line_num.2018
  %t2185 = add i64 %t2184, 1
  store i64 %t2185, ptr %local.line_num.2018
  br label %then147_end
then147_end:
  br label %endif147
else147:
  %t2186 = load i64, ptr %local.trimmed.2020
  %t2187 = call ptr @intrinsic_string_new(ptr @.str.main.124)
  %t2188 = ptrtoint ptr %t2187 to i64
  %t2189 = inttoptr i64 %t2186 to ptr
  %t2190 = inttoptr i64 %t2188 to ptr
  %t2191 = call i1 @intrinsic_string_eq(ptr %t2189, ptr %t2190)
  %t2192 = zext i1 %t2191 to i64
  %t2193 = icmp ne i64 %t2192, 0
  br i1 %t2193, label %then150, label %else150
then150:
  %t2194 = load i64, ptr %local.state.2017
  %t2195 = call ptr @intrinsic_vec_new()
  %t2196 = ptrtoint ptr %t2195 to i64
  %t2197 = inttoptr i64 %t2194 to ptr
  %t2198 = inttoptr i64 %t2196 to ptr
  %t2199 = call ptr @store_ptr(ptr %t2197, i64 0, ptr %t2198)
  %t2200 = ptrtoint ptr %t2199 to i64
  %t2201 = call ptr @intrinsic_string_new(ptr @.str.main.125)
  %t2202 = ptrtoint ptr %t2201 to i64
  %t2203 = inttoptr i64 %t2202 to ptr
  call void @intrinsic_println(ptr %t2203)
  %t2204 = load i64, ptr %local.line_num.2018
  %t2205 = add i64 %t2204, 1
  store i64 %t2205, ptr %local.line_num.2018
  br label %then150_end
then150_end:
  br label %endif150
else150:
  %t2206 = load i64, ptr %local.state.2017
  %t2207 = load i64, ptr %local.trimmed.2020
  %t2208 = call i64 @"repl_eval_line"(i64 %t2206, i64 %t2207)
  store i64 %t2208, ptr %local.result.2027
  %t2209 = load i64, ptr %local.result.2027
  %t2210 = icmp ne i64 %t2209, 0
  %t2211 = zext i1 %t2210 to i64
  %t2212 = icmp ne i64 %t2211, 0
  br i1 %t2212, label %then151, label %else151
then151:
  %t2213 = load i64, ptr %local.result.2027
  %t2214 = inttoptr i64 %t2213 to ptr
  call void @intrinsic_println(ptr %t2214)
  br label %then151_end
then151_end:
  br label %endif151
else151:
  br label %else151_end
else151_end:
  br label %endif151
endif151:
  %t2215 = phi i64 [ 0, %then151_end ], [ 0, %else151_end ]
  %t2216 = load i64, ptr %local.line_num.2018
  %t2217 = add i64 %t2216, 1
  store i64 %t2217, ptr %local.line_num.2018
  br label %else150_end
else150_end:
  br label %endif150
endif150:
  %t2218 = phi i64 [ 0, %then150_end ], [ 0, %else150_end ]
  br label %else147_end
else147_end:
  br label %endif147
endif147:
  %t2219 = phi i64 [ 0, %then147_end ], [ %t2218, %else147_end ]
  br label %else146_end
else146_end:
  br label %endif146
endif146:
  %t2220 = phi i64 [ 0, %then146_end ], [ %t2219, %else146_end ]
  br label %else145_end
else145_end:
  br label %endif145
endif145:
  %t2221 = phi i64 [ 0, %then145_end ], [ %t2220, %else145_end ]
  br label %loop141
endloop141:
  ret i64 0
}

define i64 @"repl_eval_line"(i64 %state, i64 %line) {
entry:
  %local.state = alloca i64
  store i64 %state, ptr %local.state
  %local.line = alloca i64
  store i64 %line, ptr %local.line
  %t2222 = load i64, ptr %local.line
  %t2223 = call ptr @intrinsic_string_new(ptr @.str.main.126)
  %t2224 = ptrtoint ptr %t2223 to i64
  %t2225 = inttoptr i64 %t2222 to ptr
  %t2226 = inttoptr i64 %t2224 to ptr
  %t2227 = call i64 @intrinsic_string_starts_with(ptr %t2225, ptr %t2226)
  %t2228 = icmp ne i64 %t2227, 0
  br i1 %t2228, label %then152, label %else152
then152:
  %t2229 = load i64, ptr %local.state
  %t2230 = load i64, ptr %local.line
  %t2231 = call i64 @"repl_eval_let"(i64 %t2229, i64 %t2230)
  ret i64 %t2231
  br label %then152_end
then152_end:
  br label %endif152
else152:
  br label %else152_end
else152_end:
  br label %endif152
endif152:
  %t2232 = phi i64 [ 0, %then152_end ], [ 0, %else152_end ]
  %t2233 = load i64, ptr %local.state
  %t2234 = load i64, ptr %local.line
  %t2235 = call i64 @"repl_eval_expr"(i64 %t2233, i64 %t2234)
  ret i64 %t2235
}

define i64 @"repl_eval_let"(i64 %state, i64 %line) {
entry:
  %local.source.2236 = alloca i64
  %local.tokens.2237 = alloca i64
  %local.parse_result.2238 = alloca i64
  %local.errors.2239 = alloca i64
  %local.rest.2240 = alloca i64
  %local.name.2241 = alloca i64
  %local.i.2242 = alloca i64
  %local.c.2243 = alloca i64
  %local.vars.2244 = alloca i64
  %local.pair.2245 = alloca i64
  %local.state = alloca i64
  store i64 %state, ptr %local.state
  %local.line = alloca i64
  store i64 %line, ptr %local.line
  %t2246 = call ptr @intrinsic_string_new(ptr @.str.main.127)
  %t2247 = ptrtoint ptr %t2246 to i64
  %t2248 = load i64, ptr %local.line
  %t2249 = call ptr @intrinsic_string_new(ptr @.str.main.128)
  %t2250 = ptrtoint ptr %t2249 to i64
  %t2251 = inttoptr i64 %t2248 to ptr
  %t2252 = inttoptr i64 %t2250 to ptr
  %t2253 = call ptr @intrinsic_string_concat(ptr %t2251, ptr %t2252)
  %t2254 = ptrtoint ptr %t2253 to i64
  %t2255 = inttoptr i64 %t2247 to ptr
  %t2256 = inttoptr i64 %t2254 to ptr
  %t2257 = call ptr @intrinsic_string_concat(ptr %t2255, ptr %t2256)
  %t2258 = ptrtoint ptr %t2257 to i64
  store i64 %t2258, ptr %local.source.2236
  %t2259 = load i64, ptr %local.source.2236
  %t2260 = call i64 @"tokenize"(i64 %t2259)
  store i64 %t2260, ptr %local.tokens.2237
  %t2261 = load i64, ptr %local.tokens.2237
  %t2262 = call i64 @"parse_program"(i64 %t2261)
  store i64 %t2262, ptr %local.parse_result.2238
  %t2263 = load i64, ptr %local.parse_result.2238
  %t2264 = call i64 @"parse_result_errors"(i64 %t2263)
  store i64 %t2264, ptr %local.errors.2239
  %t2265 = load i64, ptr %local.errors.2239
  %t2266 = icmp sgt i64 %t2265, 0
  %t2267 = zext i1 %t2266 to i64
  %t2268 = icmp ne i64 %t2267, 0
  br i1 %t2268, label %then153, label %else153
then153:
  %t2269 = call ptr @intrinsic_string_new(ptr @.str.main.129)
  %t2270 = ptrtoint ptr %t2269 to i64
  ret i64 %t2270
  br label %then153_end
then153_end:
  br label %endif153
else153:
  br label %else153_end
else153_end:
  br label %endif153
endif153:
  %t2271 = phi i64 [ 0, %then153_end ], [ 0, %else153_end ]
  %t2272 = load i64, ptr %local.line
  %t2273 = load i64, ptr %local.line
  %t2274 = inttoptr i64 %t2273 to ptr
  %t2275 = call i64 @intrinsic_string_len(ptr %t2274)
  %t2276 = inttoptr i64 %t2272 to ptr
  %t2277 = call ptr @intrinsic_string_slice(ptr %t2276, i64 4, i64 %t2275)
  %t2278 = ptrtoint ptr %t2277 to i64
  store i64 %t2278, ptr %local.rest.2240
  %t2279 = call ptr @intrinsic_string_new(ptr @.str.main.130)
  %t2280 = ptrtoint ptr %t2279 to i64
  store i64 %t2280, ptr %local.name.2241
  store i64 0, ptr %local.i.2242
  br label %loop154
loop154:
  %t2281 = load i64, ptr %local.i.2242
  %t2282 = load i64, ptr %local.rest.2240
  %t2283 = inttoptr i64 %t2282 to ptr
  %t2284 = call i64 @intrinsic_string_len(ptr %t2283)
  %t2285 = icmp slt i64 %t2281, %t2284
  %t2286 = zext i1 %t2285 to i64
  %t2287 = icmp ne i64 %t2286, 0
  br i1 %t2287, label %body154, label %endloop154
body154:
  %t2288 = load i64, ptr %local.rest.2240
  %t2289 = load i64, ptr %local.i.2242
  %t2290 = inttoptr i64 %t2288 to ptr
  %t2291 = call i64 @intrinsic_string_char_at(ptr %t2290, i64 %t2289)
  store i64 %t2291, ptr %local.c.2243
  %t2292 = load i64, ptr %local.c.2243
  %t2293 = icmp eq i64 %t2292, 32
  %t2294 = zext i1 %t2293 to i64
  %t2295 = icmp ne i64 %t2294, 0
  br i1 %t2295, label %then155, label %else155
then155:
  %t2296 = load i64, ptr %local.rest.2240
  %t2297 = inttoptr i64 %t2296 to ptr
  %t2298 = call i64 @intrinsic_string_len(ptr %t2297)
  store i64 %t2298, ptr %local.i.2242
  br label %then155_end
then155_end:
  br label %endif155
else155:
  %t2299 = load i64, ptr %local.c.2243
  %t2300 = icmp eq i64 %t2299, 58
  %t2301 = zext i1 %t2300 to i64
  %t2302 = icmp ne i64 %t2301, 0
  br i1 %t2302, label %then156, label %else156
then156:
  %t2303 = load i64, ptr %local.rest.2240
  %t2304 = inttoptr i64 %t2303 to ptr
  %t2305 = call i64 @intrinsic_string_len(ptr %t2304)
  store i64 %t2305, ptr %local.i.2242
  br label %then156_end
then156_end:
  br label %endif156
else156:
  %t2306 = load i64, ptr %local.c.2243
  %t2307 = icmp eq i64 %t2306, 61
  %t2308 = zext i1 %t2307 to i64
  %t2309 = icmp ne i64 %t2308, 0
  br i1 %t2309, label %then157, label %else157
then157:
  %t2310 = load i64, ptr %local.rest.2240
  %t2311 = inttoptr i64 %t2310 to ptr
  %t2312 = call i64 @intrinsic_string_len(ptr %t2311)
  store i64 %t2312, ptr %local.i.2242
  br label %then157_end
then157_end:
  br label %endif157
else157:
  %t2313 = load i64, ptr %local.name.2241
  %t2314 = load i64, ptr %local.c.2243
  %t2315 = call ptr @intrinsic_string_from_char(i64 %t2314)
  %t2316 = ptrtoint ptr %t2315 to i64
  %t2317 = inttoptr i64 %t2313 to ptr
  %t2318 = inttoptr i64 %t2316 to ptr
  %t2319 = call ptr @intrinsic_string_concat(ptr %t2317, ptr %t2318)
  %t2320 = ptrtoint ptr %t2319 to i64
  store i64 %t2320, ptr %local.name.2241
  %t2321 = load i64, ptr %local.i.2242
  %t2322 = add i64 %t2321, 1
  store i64 %t2322, ptr %local.i.2242
  br label %else157_end
else157_end:
  br label %endif157
endif157:
  %t2323 = phi i64 [ 0, %then157_end ], [ 0, %else157_end ]
  br label %else156_end
else156_end:
  br label %endif156
endif156:
  %t2324 = phi i64 [ 0, %then156_end ], [ %t2323, %else156_end ]
  br label %else155_end
else155_end:
  br label %endif155
endif155:
  %t2325 = phi i64 [ 0, %then155_end ], [ %t2324, %else155_end ]
  br label %loop154
endloop154:
  %t2326 = load i64, ptr %local.state
  %t2327 = call i64 @"repl_state_vars"(i64 %t2326)
  store i64 %t2327, ptr %local.vars.2244
  %t2328 = call ptr @malloc(i64 16)
  %t2329 = ptrtoint ptr %t2328 to i64
  store i64 %t2329, ptr %local.pair.2245
  %t2330 = load i64, ptr %local.pair.2245
  %t2331 = load i64, ptr %local.name.2241
  %t2332 = inttoptr i64 %t2330 to ptr
  %t2333 = inttoptr i64 %t2331 to ptr
  %t2334 = call ptr @store_ptr(ptr %t2332, i64 0, ptr %t2333)
  %t2335 = ptrtoint ptr %t2334 to i64
  %t2336 = load i64, ptr %local.pair.2245
  %t2337 = inttoptr i64 %t2336 to ptr
  %t2338 = call ptr @store_i64(ptr %t2337, i64 1, i64 0)
  %t2339 = ptrtoint ptr %t2338 to i64
  %t2340 = load i64, ptr %local.vars.2244
  %t2341 = load i64, ptr %local.pair.2245
  %t2342 = inttoptr i64 %t2340 to ptr
  %t2343 = inttoptr i64 %t2341 to ptr
  call void @intrinsic_vec_push(ptr %t2342, ptr %t2343)
  %t2344 = call ptr @intrinsic_string_new(ptr @.str.main.131)
  %t2345 = ptrtoint ptr %t2344 to i64
  %t2346 = load i64, ptr %local.name.2241
  %t2347 = inttoptr i64 %t2345 to ptr
  %t2348 = inttoptr i64 %t2346 to ptr
  %t2349 = call ptr @intrinsic_string_concat(ptr %t2347, ptr %t2348)
  %t2350 = ptrtoint ptr %t2349 to i64
  ret i64 %t2350
}

define i64 @"repl_eval_expr"(i64 %state, i64 %line) {
entry:
  %local.source.2351 = alloca i64
  %local.tokens.2352 = alloca i64
  %local.parse_result.2353 = alloca i64
  %local.errors.2354 = alloca i64
  %local.val.2355 = alloca i64
  %local.result.2356 = alloca i64
  %local.state = alloca i64
  store i64 %state, ptr %local.state
  %local.line = alloca i64
  store i64 %line, ptr %local.line
  %t2357 = call ptr @intrinsic_string_new(ptr @.str.main.132)
  %t2358 = ptrtoint ptr %t2357 to i64
  %t2359 = load i64, ptr %local.line
  %t2360 = call ptr @intrinsic_string_new(ptr @.str.main.133)
  %t2361 = ptrtoint ptr %t2360 to i64
  %t2362 = inttoptr i64 %t2359 to ptr
  %t2363 = inttoptr i64 %t2361 to ptr
  %t2364 = call ptr @intrinsic_string_concat(ptr %t2362, ptr %t2363)
  %t2365 = ptrtoint ptr %t2364 to i64
  %t2366 = inttoptr i64 %t2358 to ptr
  %t2367 = inttoptr i64 %t2365 to ptr
  %t2368 = call ptr @intrinsic_string_concat(ptr %t2366, ptr %t2367)
  %t2369 = ptrtoint ptr %t2368 to i64
  store i64 %t2369, ptr %local.source.2351
  %t2370 = load i64, ptr %local.source.2351
  %t2371 = call i64 @"tokenize"(i64 %t2370)
  store i64 %t2371, ptr %local.tokens.2352
  %t2372 = load i64, ptr %local.tokens.2352
  %t2373 = call i64 @"parse_program"(i64 %t2372)
  store i64 %t2373, ptr %local.parse_result.2353
  %t2374 = load i64, ptr %local.parse_result.2353
  %t2375 = call i64 @"parse_result_errors"(i64 %t2374)
  store i64 %t2375, ptr %local.errors.2354
  %t2376 = load i64, ptr %local.errors.2354
  %t2377 = icmp sgt i64 %t2376, 0
  %t2378 = zext i1 %t2377 to i64
  %t2379 = icmp ne i64 %t2378, 0
  br i1 %t2379, label %then158, label %else158
then158:
  %t2380 = call ptr @intrinsic_string_new(ptr @.str.main.134)
  %t2381 = ptrtoint ptr %t2380 to i64
  ret i64 %t2381
  br label %then158_end
then158_end:
  br label %endif158
else158:
  br label %else158_end
else158_end:
  br label %endif158
endif158:
  %t2382 = phi i64 [ 0, %then158_end ], [ 0, %else158_end ]
  %t2383 = load i64, ptr %local.line
  %t2384 = call i64 @"is_integer_literal"(i64 %t2383)
  %t2385 = icmp ne i64 %t2384, 0
  br i1 %t2385, label %then159, label %else159
then159:
  %t2386 = load i64, ptr %local.line
  %t2387 = inttoptr i64 %t2386 to ptr
  %t2388 = call i64 @intrinsic_string_to_int(ptr %t2387)
  store i64 %t2388, ptr %local.val.2355
  %t2389 = call ptr @intrinsic_string_new(ptr @.str.main.135)
  %t2390 = ptrtoint ptr %t2389 to i64
  %t2391 = load i64, ptr %local.val.2355
  %t2392 = call ptr @intrinsic_int_to_string(i64 %t2391)
  %t2393 = ptrtoint ptr %t2392 to i64
  %t2394 = inttoptr i64 %t2390 to ptr
  %t2395 = inttoptr i64 %t2393 to ptr
  %t2396 = call ptr @intrinsic_string_concat(ptr %t2394, ptr %t2395)
  %t2397 = ptrtoint ptr %t2396 to i64
  ret i64 %t2397
  br label %then159_end
then159_end:
  br label %endif159
else159:
  br label %else159_end
else159_end:
  br label %endif159
endif159:
  %t2398 = phi i64 [ 0, %then159_end ], [ 0, %else159_end ]
  %t2399 = load i64, ptr %local.line
  %t2400 = call i64 @"try_eval_simple_expr"(i64 %t2399)
  store i64 %t2400, ptr %local.result.2356
  %t2401 = load i64, ptr %local.result.2356
  %t2402 = icmp ne i64 %t2401, 0
  %t2403 = zext i1 %t2402 to i64
  %t2404 = icmp ne i64 %t2403, 0
  br i1 %t2404, label %then160, label %else160
then160:
  %t2405 = load i64, ptr %local.result.2356
  ret i64 %t2405
  br label %then160_end
then160_end:
  br label %endif160
else160:
  br label %else160_end
else160_end:
  br label %endif160
endif160:
  %t2406 = phi i64 [ 0, %then160_end ], [ 0, %else160_end ]
  %t2407 = call ptr @intrinsic_string_new(ptr @.str.main.136)
  %t2408 = ptrtoint ptr %t2407 to i64
  ret i64 %t2408
}

define i64 @"is_integer_literal"(i64 %s) {
entry:
  %local.len.2409 = alloca i64
  %local.i.2410 = alloca i64
  %local.c.2411 = alloca i64
  %local.s = alloca i64
  store i64 %s, ptr %local.s
  %t2412 = load i64, ptr %local.s
  %t2413 = inttoptr i64 %t2412 to ptr
  %t2414 = call i64 @intrinsic_string_len(ptr %t2413)
  store i64 %t2414, ptr %local.len.2409
  %t2415 = load i64, ptr %local.len.2409
  %t2416 = icmp eq i64 %t2415, 0
  %t2417 = zext i1 %t2416 to i64
  %t2418 = icmp ne i64 %t2417, 0
  br i1 %t2418, label %then161, label %else161
then161:
  ret i64 0
  br label %then161_end
then161_end:
  br label %endif161
else161:
  br label %else161_end
else161_end:
  br label %endif161
endif161:
  %t2419 = phi i64 [ 0, %then161_end ], [ 0, %else161_end ]
  store i64 0, ptr %local.i.2410
  %t2420 = load i64, ptr %local.s
  %t2421 = inttoptr i64 %t2420 to ptr
  %t2422 = call i64 @intrinsic_string_char_at(ptr %t2421, i64 0)
  %t2423 = icmp eq i64 %t2422, 45
  %t2424 = zext i1 %t2423 to i64
  %t2425 = icmp ne i64 %t2424, 0
  br i1 %t2425, label %then162, label %else162
then162:
  store i64 1, ptr %local.i.2410
  br label %then162_end
then162_end:
  br label %endif162
else162:
  br label %else162_end
else162_end:
  br label %endif162
endif162:
  %t2426 = phi i64 [ 0, %then162_end ], [ 0, %else162_end ]
  br label %loop163
loop163:
  %t2427 = load i64, ptr %local.i.2410
  %t2428 = load i64, ptr %local.len.2409
  %t2429 = icmp slt i64 %t2427, %t2428
  %t2430 = zext i1 %t2429 to i64
  %t2431 = icmp ne i64 %t2430, 0
  br i1 %t2431, label %body163, label %endloop163
body163:
  %t2432 = load i64, ptr %local.s
  %t2433 = load i64, ptr %local.i.2410
  %t2434 = inttoptr i64 %t2432 to ptr
  %t2435 = call i64 @intrinsic_string_char_at(ptr %t2434, i64 %t2433)
  store i64 %t2435, ptr %local.c.2411
  %t2436 = load i64, ptr %local.c.2411
  %t2437 = icmp slt i64 %t2436, 48
  %t2438 = zext i1 %t2437 to i64
  %t2439 = icmp ne i64 %t2438, 0
  br i1 %t2439, label %then164, label %else164
then164:
  ret i64 0
  br label %then164_end
then164_end:
  br label %endif164
else164:
  br label %else164_end
else164_end:
  br label %endif164
endif164:
  %t2440 = phi i64 [ 0, %then164_end ], [ 0, %else164_end ]
  %t2441 = load i64, ptr %local.c.2411
  %t2442 = icmp sgt i64 %t2441, 57
  %t2443 = zext i1 %t2442 to i64
  %t2444 = icmp ne i64 %t2443, 0
  br i1 %t2444, label %then165, label %else165
then165:
  ret i64 0
  br label %then165_end
then165_end:
  br label %endif165
else165:
  br label %else165_end
else165_end:
  br label %endif165
endif165:
  %t2445 = phi i64 [ 0, %then165_end ], [ 0, %else165_end ]
  %t2446 = load i64, ptr %local.i.2410
  %t2447 = add i64 %t2446, 1
  store i64 %t2447, ptr %local.i.2410
  br label %loop163
endloop163:
  ret i64 1
}

define i64 @"try_eval_simple_expr"(i64 %expr) {
entry:
  %local.len.2448 = alloca i64
  %local.i.2449 = alloca i64
  %local.c.2450 = alloca i64
  %local.left.2451 = alloca i64
  %local.right.2452 = alloca i64
  %local.result.2453 = alloca i64
  %local.left.2454 = alloca i64
  %local.right.2455 = alloca i64
  %local.result.2456 = alloca i64
  %local.left.2457 = alloca i64
  %local.right.2458 = alloca i64
  %local.result.2459 = alloca i64
  %local.left.2460 = alloca i64
  %local.right.2461 = alloca i64
  %local.rval.2462 = alloca i64
  %local.result.2463 = alloca i64
  %local.expr = alloca i64
  store i64 %expr, ptr %local.expr
  %t2464 = load i64, ptr %local.expr
  %t2465 = inttoptr i64 %t2464 to ptr
  %t2466 = call i64 @intrinsic_string_len(ptr %t2465)
  store i64 %t2466, ptr %local.len.2448
  store i64 0, ptr %local.i.2449
  br label %loop166
loop166:
  %t2467 = load i64, ptr %local.i.2449
  %t2468 = load i64, ptr %local.len.2448
  %t2469 = icmp slt i64 %t2467, %t2468
  %t2470 = zext i1 %t2469 to i64
  %t2471 = icmp ne i64 %t2470, 0
  br i1 %t2471, label %body166, label %endloop166
body166:
  %t2472 = load i64, ptr %local.expr
  %t2473 = load i64, ptr %local.i.2449
  %t2474 = inttoptr i64 %t2472 to ptr
  %t2475 = call i64 @intrinsic_string_char_at(ptr %t2474, i64 %t2473)
  store i64 %t2475, ptr %local.c.2450
  %t2476 = load i64, ptr %local.c.2450
  %t2477 = icmp eq i64 %t2476, 43
  %t2478 = zext i1 %t2477 to i64
  %t2479 = icmp ne i64 %t2478, 0
  br i1 %t2479, label %then167, label %else167
then167:
  %t2480 = load i64, ptr %local.expr
  %t2481 = load i64, ptr %local.i.2449
  %t2482 = inttoptr i64 %t2480 to ptr
  %t2483 = call ptr @intrinsic_string_slice(ptr %t2482, i64 0, i64 %t2481)
  %t2484 = ptrtoint ptr %t2483 to i64
  %t2485 = inttoptr i64 %t2484 to ptr
  %t2486 = call ptr @intrinsic_string_trim(ptr %t2485)
  %t2487 = ptrtoint ptr %t2486 to i64
  store i64 %t2487, ptr %local.left.2451
  %t2488 = load i64, ptr %local.expr
  %t2489 = load i64, ptr %local.i.2449
  %t2490 = add i64 %t2489, 1
  %t2491 = load i64, ptr %local.len.2448
  %t2492 = inttoptr i64 %t2488 to ptr
  %t2493 = call ptr @intrinsic_string_slice(ptr %t2492, i64 %t2490, i64 %t2491)
  %t2494 = ptrtoint ptr %t2493 to i64
  %t2495 = inttoptr i64 %t2494 to ptr
  %t2496 = call ptr @intrinsic_string_trim(ptr %t2495)
  %t2497 = ptrtoint ptr %t2496 to i64
  store i64 %t2497, ptr %local.right.2452
  %t2498 = load i64, ptr %local.left.2451
  %t2499 = call i64 @"is_integer_literal"(i64 %t2498)
  %t2500 = icmp ne i64 %t2499, 0
  br i1 %t2500, label %then168, label %else168
then168:
  %t2501 = load i64, ptr %local.right.2452
  %t2502 = call i64 @"is_integer_literal"(i64 %t2501)
  %t2503 = icmp ne i64 %t2502, 0
  br i1 %t2503, label %then169, label %else169
then169:
  %t2504 = load i64, ptr %local.left.2451
  %t2505 = inttoptr i64 %t2504 to ptr
  %t2506 = call i64 @intrinsic_string_to_int(ptr %t2505)
  %t2507 = load i64, ptr %local.right.2452
  %t2508 = inttoptr i64 %t2507 to ptr
  %t2509 = call i64 @intrinsic_string_to_int(ptr %t2508)
  %t2510 = add i64 %t2506, %t2509
  store i64 %t2510, ptr %local.result.2453
  %t2511 = call ptr @intrinsic_string_new(ptr @.str.main.137)
  %t2512 = ptrtoint ptr %t2511 to i64
  %t2513 = load i64, ptr %local.result.2453
  %t2514 = call ptr @intrinsic_int_to_string(i64 %t2513)
  %t2515 = ptrtoint ptr %t2514 to i64
  %t2516 = inttoptr i64 %t2512 to ptr
  %t2517 = inttoptr i64 %t2515 to ptr
  %t2518 = call ptr @intrinsic_string_concat(ptr %t2516, ptr %t2517)
  %t2519 = ptrtoint ptr %t2518 to i64
  ret i64 %t2519
  br label %then169_end
then169_end:
  br label %endif169
else169:
  br label %else169_end
else169_end:
  br label %endif169
endif169:
  %t2520 = phi i64 [ 0, %then169_end ], [ 0, %else169_end ]
  br label %then168_end
then168_end:
  br label %endif168
else168:
  br label %else168_end
else168_end:
  br label %endif168
endif168:
  %t2521 = phi i64 [ %t2520, %then168_end ], [ 0, %else168_end ]
  br label %then167_end
then167_end:
  br label %endif167
else167:
  br label %else167_end
else167_end:
  br label %endif167
endif167:
  %t2522 = phi i64 [ %t2521, %then167_end ], [ 0, %else167_end ]
  %t2523 = load i64, ptr %local.c.2450
  %t2524 = icmp eq i64 %t2523, 45
  %t2525 = zext i1 %t2524 to i64
  %t2526 = icmp ne i64 %t2525, 0
  br i1 %t2526, label %then170, label %else170
then170:
  %t2527 = load i64, ptr %local.i.2449
  %t2528 = icmp sgt i64 %t2527, 0
  %t2529 = zext i1 %t2528 to i64
  %t2530 = icmp ne i64 %t2529, 0
  br i1 %t2530, label %then171, label %else171
then171:
  %t2531 = load i64, ptr %local.expr
  %t2532 = load i64, ptr %local.i.2449
  %t2533 = inttoptr i64 %t2531 to ptr
  %t2534 = call ptr @intrinsic_string_slice(ptr %t2533, i64 0, i64 %t2532)
  %t2535 = ptrtoint ptr %t2534 to i64
  %t2536 = inttoptr i64 %t2535 to ptr
  %t2537 = call ptr @intrinsic_string_trim(ptr %t2536)
  %t2538 = ptrtoint ptr %t2537 to i64
  store i64 %t2538, ptr %local.left.2454
  %t2539 = load i64, ptr %local.expr
  %t2540 = load i64, ptr %local.i.2449
  %t2541 = add i64 %t2540, 1
  %t2542 = load i64, ptr %local.len.2448
  %t2543 = inttoptr i64 %t2539 to ptr
  %t2544 = call ptr @intrinsic_string_slice(ptr %t2543, i64 %t2541, i64 %t2542)
  %t2545 = ptrtoint ptr %t2544 to i64
  %t2546 = inttoptr i64 %t2545 to ptr
  %t2547 = call ptr @intrinsic_string_trim(ptr %t2546)
  %t2548 = ptrtoint ptr %t2547 to i64
  store i64 %t2548, ptr %local.right.2455
  %t2549 = load i64, ptr %local.left.2454
  %t2550 = call i64 @"is_integer_literal"(i64 %t2549)
  %t2551 = icmp ne i64 %t2550, 0
  br i1 %t2551, label %then172, label %else172
then172:
  %t2552 = load i64, ptr %local.right.2455
  %t2553 = call i64 @"is_integer_literal"(i64 %t2552)
  %t2554 = icmp ne i64 %t2553, 0
  br i1 %t2554, label %then173, label %else173
then173:
  %t2555 = load i64, ptr %local.left.2454
  %t2556 = inttoptr i64 %t2555 to ptr
  %t2557 = call i64 @intrinsic_string_to_int(ptr %t2556)
  %t2558 = load i64, ptr %local.right.2455
  %t2559 = inttoptr i64 %t2558 to ptr
  %t2560 = call i64 @intrinsic_string_to_int(ptr %t2559)
  %t2561 = sub i64 %t2557, %t2560
  store i64 %t2561, ptr %local.result.2456
  %t2562 = call ptr @intrinsic_string_new(ptr @.str.main.138)
  %t2563 = ptrtoint ptr %t2562 to i64
  %t2564 = load i64, ptr %local.result.2456
  %t2565 = call ptr @intrinsic_int_to_string(i64 %t2564)
  %t2566 = ptrtoint ptr %t2565 to i64
  %t2567 = inttoptr i64 %t2563 to ptr
  %t2568 = inttoptr i64 %t2566 to ptr
  %t2569 = call ptr @intrinsic_string_concat(ptr %t2567, ptr %t2568)
  %t2570 = ptrtoint ptr %t2569 to i64
  ret i64 %t2570
  br label %then173_end
then173_end:
  br label %endif173
else173:
  br label %else173_end
else173_end:
  br label %endif173
endif173:
  %t2571 = phi i64 [ 0, %then173_end ], [ 0, %else173_end ]
  br label %then172_end
then172_end:
  br label %endif172
else172:
  br label %else172_end
else172_end:
  br label %endif172
endif172:
  %t2572 = phi i64 [ %t2571, %then172_end ], [ 0, %else172_end ]
  br label %then171_end
then171_end:
  br label %endif171
else171:
  br label %else171_end
else171_end:
  br label %endif171
endif171:
  %t2573 = phi i64 [ %t2572, %then171_end ], [ 0, %else171_end ]
  br label %then170_end
then170_end:
  br label %endif170
else170:
  br label %else170_end
else170_end:
  br label %endif170
endif170:
  %t2574 = phi i64 [ %t2573, %then170_end ], [ 0, %else170_end ]
  %t2575 = load i64, ptr %local.c.2450
  %t2576 = icmp eq i64 %t2575, 42
  %t2577 = zext i1 %t2576 to i64
  %t2578 = icmp ne i64 %t2577, 0
  br i1 %t2578, label %then174, label %else174
then174:
  %t2579 = load i64, ptr %local.expr
  %t2580 = load i64, ptr %local.i.2449
  %t2581 = inttoptr i64 %t2579 to ptr
  %t2582 = call ptr @intrinsic_string_slice(ptr %t2581, i64 0, i64 %t2580)
  %t2583 = ptrtoint ptr %t2582 to i64
  %t2584 = inttoptr i64 %t2583 to ptr
  %t2585 = call ptr @intrinsic_string_trim(ptr %t2584)
  %t2586 = ptrtoint ptr %t2585 to i64
  store i64 %t2586, ptr %local.left.2457
  %t2587 = load i64, ptr %local.expr
  %t2588 = load i64, ptr %local.i.2449
  %t2589 = add i64 %t2588, 1
  %t2590 = load i64, ptr %local.len.2448
  %t2591 = inttoptr i64 %t2587 to ptr
  %t2592 = call ptr @intrinsic_string_slice(ptr %t2591, i64 %t2589, i64 %t2590)
  %t2593 = ptrtoint ptr %t2592 to i64
  %t2594 = inttoptr i64 %t2593 to ptr
  %t2595 = call ptr @intrinsic_string_trim(ptr %t2594)
  %t2596 = ptrtoint ptr %t2595 to i64
  store i64 %t2596, ptr %local.right.2458
  %t2597 = load i64, ptr %local.left.2457
  %t2598 = call i64 @"is_integer_literal"(i64 %t2597)
  %t2599 = icmp ne i64 %t2598, 0
  br i1 %t2599, label %then175, label %else175
then175:
  %t2600 = load i64, ptr %local.right.2458
  %t2601 = call i64 @"is_integer_literal"(i64 %t2600)
  %t2602 = icmp ne i64 %t2601, 0
  br i1 %t2602, label %then176, label %else176
then176:
  %t2603 = load i64, ptr %local.left.2457
  %t2604 = inttoptr i64 %t2603 to ptr
  %t2605 = call i64 @intrinsic_string_to_int(ptr %t2604)
  %t2606 = load i64, ptr %local.right.2458
  %t2607 = inttoptr i64 %t2606 to ptr
  %t2608 = call i64 @intrinsic_string_to_int(ptr %t2607)
  %t2609 = mul i64 %t2605, %t2608
  store i64 %t2609, ptr %local.result.2459
  %t2610 = call ptr @intrinsic_string_new(ptr @.str.main.139)
  %t2611 = ptrtoint ptr %t2610 to i64
  %t2612 = load i64, ptr %local.result.2459
  %t2613 = call ptr @intrinsic_int_to_string(i64 %t2612)
  %t2614 = ptrtoint ptr %t2613 to i64
  %t2615 = inttoptr i64 %t2611 to ptr
  %t2616 = inttoptr i64 %t2614 to ptr
  %t2617 = call ptr @intrinsic_string_concat(ptr %t2615, ptr %t2616)
  %t2618 = ptrtoint ptr %t2617 to i64
  ret i64 %t2618
  br label %then176_end
then176_end:
  br label %endif176
else176:
  br label %else176_end
else176_end:
  br label %endif176
endif176:
  %t2619 = phi i64 [ 0, %then176_end ], [ 0, %else176_end ]
  br label %then175_end
then175_end:
  br label %endif175
else175:
  br label %else175_end
else175_end:
  br label %endif175
endif175:
  %t2620 = phi i64 [ %t2619, %then175_end ], [ 0, %else175_end ]
  br label %then174_end
then174_end:
  br label %endif174
else174:
  br label %else174_end
else174_end:
  br label %endif174
endif174:
  %t2621 = phi i64 [ %t2620, %then174_end ], [ 0, %else174_end ]
  %t2622 = load i64, ptr %local.c.2450
  %t2623 = icmp eq i64 %t2622, 47
  %t2624 = zext i1 %t2623 to i64
  %t2625 = icmp ne i64 %t2624, 0
  br i1 %t2625, label %then177, label %else177
then177:
  %t2626 = load i64, ptr %local.expr
  %t2627 = load i64, ptr %local.i.2449
  %t2628 = inttoptr i64 %t2626 to ptr
  %t2629 = call ptr @intrinsic_string_slice(ptr %t2628, i64 0, i64 %t2627)
  %t2630 = ptrtoint ptr %t2629 to i64
  %t2631 = inttoptr i64 %t2630 to ptr
  %t2632 = call ptr @intrinsic_string_trim(ptr %t2631)
  %t2633 = ptrtoint ptr %t2632 to i64
  store i64 %t2633, ptr %local.left.2460
  %t2634 = load i64, ptr %local.expr
  %t2635 = load i64, ptr %local.i.2449
  %t2636 = add i64 %t2635, 1
  %t2637 = load i64, ptr %local.len.2448
  %t2638 = inttoptr i64 %t2634 to ptr
  %t2639 = call ptr @intrinsic_string_slice(ptr %t2638, i64 %t2636, i64 %t2637)
  %t2640 = ptrtoint ptr %t2639 to i64
  %t2641 = inttoptr i64 %t2640 to ptr
  %t2642 = call ptr @intrinsic_string_trim(ptr %t2641)
  %t2643 = ptrtoint ptr %t2642 to i64
  store i64 %t2643, ptr %local.right.2461
  %t2644 = load i64, ptr %local.left.2460
  %t2645 = call i64 @"is_integer_literal"(i64 %t2644)
  %t2646 = icmp ne i64 %t2645, 0
  br i1 %t2646, label %then178, label %else178
then178:
  %t2647 = load i64, ptr %local.right.2461
  %t2648 = call i64 @"is_integer_literal"(i64 %t2647)
  %t2649 = icmp ne i64 %t2648, 0
  br i1 %t2649, label %then179, label %else179
then179:
  %t2650 = load i64, ptr %local.right.2461
  %t2651 = inttoptr i64 %t2650 to ptr
  %t2652 = call i64 @intrinsic_string_to_int(ptr %t2651)
  store i64 %t2652, ptr %local.rval.2462
  %t2653 = load i64, ptr %local.rval.2462
  %t2654 = icmp ne i64 %t2653, 0
  %t2655 = zext i1 %t2654 to i64
  %t2656 = icmp ne i64 %t2655, 0
  br i1 %t2656, label %then180, label %else180
then180:
  %t2657 = load i64, ptr %local.left.2460
  %t2658 = inttoptr i64 %t2657 to ptr
  %t2659 = call i64 @intrinsic_string_to_int(ptr %t2658)
  %t2660 = load i64, ptr %local.rval.2462
  %t2661 = sdiv i64 %t2659, %t2660
  store i64 %t2661, ptr %local.result.2463
  %t2662 = call ptr @intrinsic_string_new(ptr @.str.main.140)
  %t2663 = ptrtoint ptr %t2662 to i64
  %t2664 = load i64, ptr %local.result.2463
  %t2665 = call ptr @intrinsic_int_to_string(i64 %t2664)
  %t2666 = ptrtoint ptr %t2665 to i64
  %t2667 = inttoptr i64 %t2663 to ptr
  %t2668 = inttoptr i64 %t2666 to ptr
  %t2669 = call ptr @intrinsic_string_concat(ptr %t2667, ptr %t2668)
  %t2670 = ptrtoint ptr %t2669 to i64
  ret i64 %t2670
  br label %then180_end
then180_end:
  br label %endif180
else180:
  %t2671 = call ptr @intrinsic_string_new(ptr @.str.main.141)
  %t2672 = ptrtoint ptr %t2671 to i64
  ret i64 %t2672
  br label %else180_end
else180_end:
  br label %endif180
endif180:
  %t2673 = phi i64 [ 0, %then180_end ], [ 0, %else180_end ]
  br label %then179_end
then179_end:
  br label %endif179
else179:
  br label %else179_end
else179_end:
  br label %endif179
endif179:
  %t2674 = phi i64 [ %t2673, %then179_end ], [ 0, %else179_end ]
  br label %then178_end
then178_end:
  br label %endif178
else178:
  br label %else178_end
else178_end:
  br label %endif178
endif178:
  %t2675 = phi i64 [ %t2674, %then178_end ], [ 0, %else178_end ]
  br label %then177_end
then177_end:
  br label %endif177
else177:
  br label %else177_end
else177_end:
  br label %endif177
endif177:
  %t2676 = phi i64 [ %t2675, %then177_end ], [ 0, %else177_end ]
  %t2677 = load i64, ptr %local.i.2449
  %t2678 = add i64 %t2677, 1
  store i64 %t2678, ptr %local.i.2449
  br label %loop166
endloop166:
  ret i64 0
}

define i64 @"format_file"(i64 %path, i64 %write_back) {
entry:
  %local.source.2679 = alloca i64
  %local.tokens.2680 = alloca i64
  %local.parse_result.2681 = alloca i64
  %local.items.2682 = alloca i64
  %local.errors.2683 = alloca i64
  %local.formatted.2684 = alloca i64
  %local.path = alloca i64
  store i64 %path, ptr %local.path
  %local.write_back = alloca i64
  store i64 %write_back, ptr %local.write_back
  %t2685 = load i64, ptr %local.path
  %t2686 = inttoptr i64 %t2685 to ptr
  %t2687 = call ptr @intrinsic_read_file(ptr %t2686)
  %t2688 = ptrtoint ptr %t2687 to i64
  store i64 %t2688, ptr %local.source.2679
  %t2689 = load i64, ptr %local.source.2679
  %t2690 = icmp eq i64 %t2689, 0
  %t2691 = zext i1 %t2690 to i64
  %t2692 = icmp ne i64 %t2691, 0
  br i1 %t2692, label %then181, label %else181
then181:
  %t2693 = call i64 @"E_CANNOT_READ_FILE"()
  %t2694 = call ptr @intrinsic_string_new(ptr @.str.main.142)
  %t2695 = ptrtoint ptr %t2694 to i64
  %t2696 = load i64, ptr %local.path
  %t2697 = inttoptr i64 %t2695 to ptr
  %t2698 = inttoptr i64 %t2696 to ptr
  %t2699 = call ptr @intrinsic_string_concat(ptr %t2697, ptr %t2698)
  %t2700 = ptrtoint ptr %t2699 to i64
  %t2701 = call i64 @"report_simple_error"(i64 %t2693, i64 %t2700)
  ret i64 1
  br label %then181_end
then181_end:
  br label %endif181
else181:
  br label %else181_end
else181_end:
  br label %endif181
endif181:
  %t2702 = phi i64 [ 0, %then181_end ], [ 0, %else181_end ]
  %t2703 = load i64, ptr %local.source.2679
  %t2704 = call i64 @"tokenize"(i64 %t2703)
  store i64 %t2704, ptr %local.tokens.2680
  %t2705 = load i64, ptr %local.tokens.2680
  %t2706 = load i64, ptr %local.source.2679
  %t2707 = load i64, ptr %local.path
  %t2708 = call i64 @"parse_program_with_source"(i64 %t2705, i64 %t2706, i64 %t2707)
  store i64 %t2708, ptr %local.parse_result.2681
  %t2709 = load i64, ptr %local.parse_result.2681
  %t2710 = call i64 @"parse_result_items"(i64 %t2709)
  store i64 %t2710, ptr %local.items.2682
  %t2711 = load i64, ptr %local.parse_result.2681
  %t2712 = call i64 @"parse_result_errors"(i64 %t2711)
  store i64 %t2712, ptr %local.errors.2683
  %t2713 = load i64, ptr %local.errors.2683
  %t2714 = icmp sgt i64 %t2713, 0
  %t2715 = zext i1 %t2714 to i64
  %t2716 = icmp ne i64 %t2715, 0
  br i1 %t2716, label %then182, label %else182
then182:
  %t2717 = load i64, ptr %local.errors.2683
  %t2718 = call i64 @"print_error_summary"(i64 %t2717, i64 0)
  ret i64 1
  br label %then182_end
then182_end:
  br label %endif182
else182:
  br label %else182_end
else182_end:
  br label %endif182
endif182:
  %t2719 = phi i64 [ 0, %then182_end ], [ 0, %else182_end ]
  %t2720 = load i64, ptr %local.items.2682
  %t2721 = call i64 @"format_items"(i64 %t2720)
  store i64 %t2721, ptr %local.formatted.2684
  %t2722 = load i64, ptr %local.write_back
  %t2723 = icmp eq i64 %t2722, 1
  %t2724 = zext i1 %t2723 to i64
  %t2725 = icmp ne i64 %t2724, 0
  br i1 %t2725, label %then183, label %else183
then183:
  %t2726 = load i64, ptr %local.path
  %t2727 = load i64, ptr %local.formatted.2684
  %t2728 = inttoptr i64 %t2726 to ptr
  %t2729 = inttoptr i64 %t2727 to ptr
  call void @intrinsic_write_file(ptr %t2728, ptr %t2729)
  %t2730 = call ptr @intrinsic_string_new(ptr @.str.main.143)
  %t2731 = ptrtoint ptr %t2730 to i64
  %t2732 = load i64, ptr %local.path
  %t2733 = inttoptr i64 %t2731 to ptr
  %t2734 = inttoptr i64 %t2732 to ptr
  %t2735 = call ptr @intrinsic_string_concat(ptr %t2733, ptr %t2734)
  %t2736 = ptrtoint ptr %t2735 to i64
  %t2737 = inttoptr i64 %t2736 to ptr
  call void @intrinsic_println(ptr %t2737)
  br label %then183_end
then183_end:
  br label %endif183
else183:
  %t2738 = load i64, ptr %local.formatted.2684
  %t2739 = inttoptr i64 %t2738 to ptr
  call void @intrinsic_println(ptr %t2739)
  br label %else183_end
else183_end:
  br label %endif183
endif183:
  %t2740 = phi i64 [ 0, %then183_end ], [ 0, %else183_end ]
  ret i64 0
}

define i64 @"format_items"(i64 %items) {
entry:
  %local.sb.2741 = alloca i64
  %local.n.2742 = alloca i64
  %local.i.2743 = alloca i64
  %local.item.2744 = alloca i64
  %local.formatted_item.2745 = alloca i64
  %local.items = alloca i64
  store i64 %items, ptr %local.items
  %t2746 = call ptr @intrinsic_sb_new()
  %t2747 = ptrtoint ptr %t2746 to i64
  store i64 %t2747, ptr %local.sb.2741
  %t2748 = load i64, ptr %local.items
  %t2749 = inttoptr i64 %t2748 to ptr
  %t2750 = call i64 @intrinsic_vec_len(ptr %t2749)
  store i64 %t2750, ptr %local.n.2742
  store i64 0, ptr %local.i.2743
  br label %loop184
loop184:
  %t2751 = load i64, ptr %local.i.2743
  %t2752 = load i64, ptr %local.n.2742
  %t2753 = icmp slt i64 %t2751, %t2752
  %t2754 = zext i1 %t2753 to i64
  %t2755 = icmp ne i64 %t2754, 0
  br i1 %t2755, label %body184, label %endloop184
body184:
  %t2756 = load i64, ptr %local.items
  %t2757 = load i64, ptr %local.i.2743
  %t2758 = inttoptr i64 %t2756 to ptr
  %t2759 = call ptr @intrinsic_vec_get(ptr %t2758, i64 %t2757)
  %t2760 = ptrtoint ptr %t2759 to i64
  store i64 %t2760, ptr %local.item.2744
  %t2761 = load i64, ptr %local.item.2744
  %t2762 = call i64 @"format_item"(i64 %t2761)
  store i64 %t2762, ptr %local.formatted_item.2745
  %t2763 = load i64, ptr %local.sb.2741
  %t2764 = load i64, ptr %local.formatted_item.2745
  %t2765 = inttoptr i64 %t2763 to ptr
  %t2766 = inttoptr i64 %t2764 to ptr
  call void @intrinsic_sb_append(ptr %t2765, ptr %t2766)
  %t2767 = load i64, ptr %local.sb.2741
  %t2768 = call ptr @intrinsic_string_new(ptr @.str.main.144)
  %t2769 = ptrtoint ptr %t2768 to i64
  %t2770 = inttoptr i64 %t2767 to ptr
  %t2771 = inttoptr i64 %t2769 to ptr
  call void @intrinsic_sb_append(ptr %t2770, ptr %t2771)
  %t2772 = load i64, ptr %local.i.2743
  %t2773 = add i64 %t2772, 1
  store i64 %t2773, ptr %local.i.2743
  br label %loop184
endloop184:
  %t2774 = load i64, ptr %local.sb.2741
  %t2775 = inttoptr i64 %t2774 to ptr
  %t2776 = call ptr @intrinsic_sb_to_string(ptr %t2775)
  %t2777 = ptrtoint ptr %t2776 to i64
  ret i64 %t2777
}

define i64 @"format_item"(i64 %item) {
entry:
  %local.tag.2778 = alloca i64
  %local.item = alloca i64
  store i64 %item, ptr %local.item
  %t2779 = load i64, ptr %local.item
  %t2780 = inttoptr i64 %t2779 to ptr
  %t2781 = call i64 @load_i64(ptr %t2780, i64 0)
  store i64 %t2781, ptr %local.tag.2778
  %t2782 = load i64, ptr %local.tag.2778
  %t2783 = icmp eq i64 %t2782, 0
  %t2784 = zext i1 %t2783 to i64
  %t2785 = icmp ne i64 %t2784, 0
  br i1 %t2785, label %then185, label %else185
then185:
  %t2786 = load i64, ptr %local.item
  %t2787 = call i64 @"format_fn"(i64 %t2786)
  ret i64 %t2787
  br label %then185_end
then185_end:
  br label %endif185
else185:
  br label %else185_end
else185_end:
  br label %endif185
endif185:
  %t2788 = phi i64 [ 0, %then185_end ], [ 0, %else185_end ]
  %t2789 = load i64, ptr %local.tag.2778
  %t2790 = icmp eq i64 %t2789, 1
  %t2791 = zext i1 %t2790 to i64
  %t2792 = icmp ne i64 %t2791, 0
  br i1 %t2792, label %then186, label %else186
then186:
  %t2793 = load i64, ptr %local.item
  %t2794 = call i64 @"format_enum"(i64 %t2793)
  ret i64 %t2794
  br label %then186_end
then186_end:
  br label %endif186
else186:
  br label %else186_end
else186_end:
  br label %endif186
endif186:
  %t2795 = phi i64 [ 0, %then186_end ], [ 0, %else186_end ]
  %t2796 = load i64, ptr %local.tag.2778
  %t2797 = icmp eq i64 %t2796, 2
  %t2798 = zext i1 %t2797 to i64
  %t2799 = icmp ne i64 %t2798, 0
  br i1 %t2799, label %then187, label %else187
then187:
  %t2800 = load i64, ptr %local.item
  %t2801 = call i64 @"format_struct"(i64 %t2800)
  ret i64 %t2801
  br label %then187_end
then187_end:
  br label %endif187
else187:
  br label %else187_end
else187_end:
  br label %endif187
endif187:
  %t2802 = phi i64 [ 0, %then187_end ], [ 0, %else187_end ]
  %t2803 = load i64, ptr %local.tag.2778
  %t2804 = icmp eq i64 %t2803, 3
  %t2805 = zext i1 %t2804 to i64
  %t2806 = icmp ne i64 %t2805, 0
  br i1 %t2806, label %then188, label %else188
then188:
  %t2807 = load i64, ptr %local.item
  %t2808 = call i64 @"format_impl"(i64 %t2807)
  ret i64 %t2808
  br label %then188_end
then188_end:
  br label %endif188
else188:
  br label %else188_end
else188_end:
  br label %endif188
endif188:
  %t2809 = phi i64 [ 0, %then188_end ], [ 0, %else188_end ]
  %t2810 = load i64, ptr %local.tag.2778
  %t2811 = icmp eq i64 %t2810, 4
  %t2812 = zext i1 %t2811 to i64
  %t2813 = icmp ne i64 %t2812, 0
  br i1 %t2813, label %then189, label %else189
then189:
  %t2814 = load i64, ptr %local.item
  %t2815 = call i64 @"format_trait"(i64 %t2814)
  ret i64 %t2815
  br label %then189_end
then189_end:
  br label %endif189
else189:
  br label %else189_end
else189_end:
  br label %endif189
endif189:
  %t2816 = phi i64 [ 0, %then189_end ], [ 0, %else189_end ]
  %t2817 = load i64, ptr %local.tag.2778
  %t2818 = icmp eq i64 %t2817, 5
  %t2819 = zext i1 %t2818 to i64
  %t2820 = icmp ne i64 %t2819, 0
  br i1 %t2820, label %then190, label %else190
then190:
  %t2821 = load i64, ptr %local.item
  %t2822 = call i64 @"format_impl_trait"(i64 %t2821)
  ret i64 %t2822
  br label %then190_end
then190_end:
  br label %endif190
else190:
  br label %else190_end
else190_end:
  br label %endif190
endif190:
  %t2823 = phi i64 [ 0, %then190_end ], [ 0, %else190_end ]
  %t2824 = call ptr @intrinsic_string_new(ptr @.str.main.145)
  %t2825 = ptrtoint ptr %t2824 to i64
  ret i64 %t2825
}

define i64 @"format_fn"(i64 %fn_def) {
entry:
  %local.sb.2826 = alloca i64
  %local.name.2827 = alloca i64
  %local.type_params.2828 = alloca i64
  %local.params.2829 = alloca i64
  %local.ret_ty.2830 = alloca i64
  %local.body.2831 = alloca i64
  %local.i.2832 = alloca i64
  %local.i.2833 = alloca i64
  %local.param.2834 = alloca i64
  %local.pname.2835 = alloca i64
  %local.pty.2836 = alloca i64
  %local.fn_def = alloca i64
  store i64 %fn_def, ptr %local.fn_def
  %t2837 = call ptr @intrinsic_sb_new()
  %t2838 = ptrtoint ptr %t2837 to i64
  store i64 %t2838, ptr %local.sb.2826
  %t2839 = load i64, ptr %local.fn_def
  %t2840 = inttoptr i64 %t2839 to ptr
  %t2841 = call ptr @load_ptr(ptr %t2840, i64 1)
  %t2842 = ptrtoint ptr %t2841 to i64
  store i64 %t2842, ptr %local.name.2827
  %t2843 = load i64, ptr %local.fn_def
  %t2844 = inttoptr i64 %t2843 to ptr
  %t2845 = call ptr @load_ptr(ptr %t2844, i64 2)
  %t2846 = ptrtoint ptr %t2845 to i64
  store i64 %t2846, ptr %local.type_params.2828
  %t2847 = load i64, ptr %local.fn_def
  %t2848 = inttoptr i64 %t2847 to ptr
  %t2849 = call ptr @load_ptr(ptr %t2848, i64 3)
  %t2850 = ptrtoint ptr %t2849 to i64
  store i64 %t2850, ptr %local.params.2829
  %t2851 = load i64, ptr %local.fn_def
  %t2852 = inttoptr i64 %t2851 to ptr
  %t2853 = call ptr @load_ptr(ptr %t2852, i64 4)
  %t2854 = ptrtoint ptr %t2853 to i64
  store i64 %t2854, ptr %local.ret_ty.2830
  %t2855 = load i64, ptr %local.fn_def
  %t2856 = inttoptr i64 %t2855 to ptr
  %t2857 = call ptr @load_ptr(ptr %t2856, i64 5)
  %t2858 = ptrtoint ptr %t2857 to i64
  store i64 %t2858, ptr %local.body.2831
  %t2859 = load i64, ptr %local.sb.2826
  %t2860 = call ptr @intrinsic_string_new(ptr @.str.main.146)
  %t2861 = ptrtoint ptr %t2860 to i64
  %t2862 = inttoptr i64 %t2859 to ptr
  %t2863 = inttoptr i64 %t2861 to ptr
  call void @intrinsic_sb_append(ptr %t2862, ptr %t2863)
  %t2864 = load i64, ptr %local.sb.2826
  %t2865 = load i64, ptr %local.name.2827
  %t2866 = inttoptr i64 %t2864 to ptr
  %t2867 = inttoptr i64 %t2865 to ptr
  call void @intrinsic_sb_append(ptr %t2866, ptr %t2867)
  %t2868 = load i64, ptr %local.type_params.2828
  %t2869 = icmp ne i64 %t2868, 0
  %t2870 = zext i1 %t2869 to i64
  %t2871 = icmp ne i64 %t2870, 0
  br i1 %t2871, label %then191, label %else191
then191:
  %t2872 = load i64, ptr %local.type_params.2828
  %t2873 = inttoptr i64 %t2872 to ptr
  %t2874 = call i64 @intrinsic_vec_len(ptr %t2873)
  %t2875 = icmp sgt i64 %t2874, 0
  %t2876 = zext i1 %t2875 to i64
  %t2877 = icmp ne i64 %t2876, 0
  br i1 %t2877, label %then192, label %else192
then192:
  %t2878 = load i64, ptr %local.sb.2826
  %t2879 = call ptr @intrinsic_string_new(ptr @.str.main.147)
  %t2880 = ptrtoint ptr %t2879 to i64
  %t2881 = inttoptr i64 %t2878 to ptr
  %t2882 = inttoptr i64 %t2880 to ptr
  call void @intrinsic_sb_append(ptr %t2881, ptr %t2882)
  store i64 0, ptr %local.i.2832
  br label %loop193
loop193:
  %t2883 = load i64, ptr %local.i.2832
  %t2884 = load i64, ptr %local.type_params.2828
  %t2885 = inttoptr i64 %t2884 to ptr
  %t2886 = call i64 @intrinsic_vec_len(ptr %t2885)
  %t2887 = icmp slt i64 %t2883, %t2886
  %t2888 = zext i1 %t2887 to i64
  %t2889 = icmp ne i64 %t2888, 0
  br i1 %t2889, label %body193, label %endloop193
body193:
  %t2890 = load i64, ptr %local.i.2832
  %t2891 = icmp sgt i64 %t2890, 0
  %t2892 = zext i1 %t2891 to i64
  %t2893 = icmp ne i64 %t2892, 0
  br i1 %t2893, label %then194, label %else194
then194:
  %t2894 = load i64, ptr %local.sb.2826
  %t2895 = call ptr @intrinsic_string_new(ptr @.str.main.148)
  %t2896 = ptrtoint ptr %t2895 to i64
  %t2897 = inttoptr i64 %t2894 to ptr
  %t2898 = inttoptr i64 %t2896 to ptr
  call void @intrinsic_sb_append(ptr %t2897, ptr %t2898)
  br label %then194_end
then194_end:
  br label %endif194
else194:
  br label %else194_end
else194_end:
  br label %endif194
endif194:
  %t2899 = phi i64 [ 0, %then194_end ], [ 0, %else194_end ]
  %t2900 = load i64, ptr %local.sb.2826
  %t2901 = load i64, ptr %local.type_params.2828
  %t2902 = load i64, ptr %local.i.2832
  %t2903 = inttoptr i64 %t2901 to ptr
  %t2904 = call ptr @intrinsic_vec_get(ptr %t2903, i64 %t2902)
  %t2905 = ptrtoint ptr %t2904 to i64
  %t2906 = inttoptr i64 %t2900 to ptr
  %t2907 = inttoptr i64 %t2905 to ptr
  call void @intrinsic_sb_append(ptr %t2906, ptr %t2907)
  %t2908 = load i64, ptr %local.i.2832
  %t2909 = add i64 %t2908, 1
  store i64 %t2909, ptr %local.i.2832
  br label %loop193
endloop193:
  %t2910 = load i64, ptr %local.sb.2826
  %t2911 = call ptr @intrinsic_string_new(ptr @.str.main.149)
  %t2912 = ptrtoint ptr %t2911 to i64
  %t2913 = inttoptr i64 %t2910 to ptr
  %t2914 = inttoptr i64 %t2912 to ptr
  call void @intrinsic_sb_append(ptr %t2913, ptr %t2914)
  br label %then192_end
then192_end:
  br label %endif192
else192:
  br label %else192_end
else192_end:
  br label %endif192
endif192:
  %t2915 = phi i64 [ 0, %then192_end ], [ 0, %else192_end ]
  br label %then191_end
then191_end:
  br label %endif191
else191:
  br label %else191_end
else191_end:
  br label %endif191
endif191:
  %t2916 = phi i64 [ %t2915, %then191_end ], [ 0, %else191_end ]
  %t2917 = load i64, ptr %local.sb.2826
  %t2918 = call ptr @intrinsic_string_new(ptr @.str.main.150)
  %t2919 = ptrtoint ptr %t2918 to i64
  %t2920 = inttoptr i64 %t2917 to ptr
  %t2921 = inttoptr i64 %t2919 to ptr
  call void @intrinsic_sb_append(ptr %t2920, ptr %t2921)
  store i64 0, ptr %local.i.2833
  br label %loop195
loop195:
  %t2922 = load i64, ptr %local.i.2833
  %t2923 = load i64, ptr %local.params.2829
  %t2924 = inttoptr i64 %t2923 to ptr
  %t2925 = call i64 @intrinsic_vec_len(ptr %t2924)
  %t2926 = icmp slt i64 %t2922, %t2925
  %t2927 = zext i1 %t2926 to i64
  %t2928 = icmp ne i64 %t2927, 0
  br i1 %t2928, label %body195, label %endloop195
body195:
  %t2929 = load i64, ptr %local.i.2833
  %t2930 = icmp sgt i64 %t2929, 0
  %t2931 = zext i1 %t2930 to i64
  %t2932 = icmp ne i64 %t2931, 0
  br i1 %t2932, label %then196, label %else196
then196:
  %t2933 = load i64, ptr %local.sb.2826
  %t2934 = call ptr @intrinsic_string_new(ptr @.str.main.151)
  %t2935 = ptrtoint ptr %t2934 to i64
  %t2936 = inttoptr i64 %t2933 to ptr
  %t2937 = inttoptr i64 %t2935 to ptr
  call void @intrinsic_sb_append(ptr %t2936, ptr %t2937)
  br label %then196_end
then196_end:
  br label %endif196
else196:
  br label %else196_end
else196_end:
  br label %endif196
endif196:
  %t2938 = phi i64 [ 0, %then196_end ], [ 0, %else196_end ]
  %t2939 = load i64, ptr %local.params.2829
  %t2940 = load i64, ptr %local.i.2833
  %t2941 = inttoptr i64 %t2939 to ptr
  %t2942 = call ptr @intrinsic_vec_get(ptr %t2941, i64 %t2940)
  %t2943 = ptrtoint ptr %t2942 to i64
  store i64 %t2943, ptr %local.param.2834
  %t2944 = load i64, ptr %local.param.2834
  %t2945 = inttoptr i64 %t2944 to ptr
  %t2946 = call ptr @load_ptr(ptr %t2945, i64 0)
  %t2947 = ptrtoint ptr %t2946 to i64
  store i64 %t2947, ptr %local.pname.2835
  %t2948 = load i64, ptr %local.param.2834
  %t2949 = inttoptr i64 %t2948 to ptr
  %t2950 = call ptr @load_ptr(ptr %t2949, i64 1)
  %t2951 = ptrtoint ptr %t2950 to i64
  store i64 %t2951, ptr %local.pty.2836
  %t2952 = load i64, ptr %local.sb.2826
  %t2953 = load i64, ptr %local.pname.2835
  %t2954 = inttoptr i64 %t2952 to ptr
  %t2955 = inttoptr i64 %t2953 to ptr
  call void @intrinsic_sb_append(ptr %t2954, ptr %t2955)
  %t2956 = load i64, ptr %local.sb.2826
  %t2957 = call ptr @intrinsic_string_new(ptr @.str.main.152)
  %t2958 = ptrtoint ptr %t2957 to i64
  %t2959 = inttoptr i64 %t2956 to ptr
  %t2960 = inttoptr i64 %t2958 to ptr
  call void @intrinsic_sb_append(ptr %t2959, ptr %t2960)
  %t2961 = load i64, ptr %local.sb.2826
  %t2962 = load i64, ptr %local.pty.2836
  %t2963 = inttoptr i64 %t2961 to ptr
  %t2964 = inttoptr i64 %t2962 to ptr
  call void @intrinsic_sb_append(ptr %t2963, ptr %t2964)
  %t2965 = load i64, ptr %local.i.2833
  %t2966 = add i64 %t2965, 1
  store i64 %t2966, ptr %local.i.2833
  br label %loop195
endloop195:
  %t2967 = load i64, ptr %local.sb.2826
  %t2968 = call ptr @intrinsic_string_new(ptr @.str.main.153)
  %t2969 = ptrtoint ptr %t2968 to i64
  %t2970 = inttoptr i64 %t2967 to ptr
  %t2971 = inttoptr i64 %t2969 to ptr
  call void @intrinsic_sb_append(ptr %t2970, ptr %t2971)
  %t2972 = load i64, ptr %local.ret_ty.2830
  %t2973 = icmp ne i64 %t2972, 0
  %t2974 = zext i1 %t2973 to i64
  %t2975 = icmp ne i64 %t2974, 0
  br i1 %t2975, label %then197, label %else197
then197:
  %t2976 = load i64, ptr %local.sb.2826
  %t2977 = call ptr @intrinsic_string_new(ptr @.str.main.154)
  %t2978 = ptrtoint ptr %t2977 to i64
  %t2979 = inttoptr i64 %t2976 to ptr
  %t2980 = inttoptr i64 %t2978 to ptr
  call void @intrinsic_sb_append(ptr %t2979, ptr %t2980)
  %t2981 = load i64, ptr %local.sb.2826
  %t2982 = load i64, ptr %local.ret_ty.2830
  %t2983 = inttoptr i64 %t2981 to ptr
  %t2984 = inttoptr i64 %t2982 to ptr
  call void @intrinsic_sb_append(ptr %t2983, ptr %t2984)
  br label %then197_end
then197_end:
  br label %endif197
else197:
  br label %else197_end
else197_end:
  br label %endif197
endif197:
  %t2985 = phi i64 [ 0, %then197_end ], [ 0, %else197_end ]
  %t2986 = load i64, ptr %local.sb.2826
  %t2987 = call ptr @intrinsic_string_new(ptr @.str.main.155)
  %t2988 = ptrtoint ptr %t2987 to i64
  %t2989 = inttoptr i64 %t2986 to ptr
  %t2990 = inttoptr i64 %t2988 to ptr
  call void @intrinsic_sb_append(ptr %t2989, ptr %t2990)
  %t2991 = load i64, ptr %local.sb.2826
  %t2992 = load i64, ptr %local.body.2831
  %t2993 = call i64 @"format_block"(i64 %t2992, i64 0)
  %t2994 = inttoptr i64 %t2991 to ptr
  %t2995 = inttoptr i64 %t2993 to ptr
  call void @intrinsic_sb_append(ptr %t2994, ptr %t2995)
  %t2996 = load i64, ptr %local.sb.2826
  %t2997 = inttoptr i64 %t2996 to ptr
  %t2998 = call ptr @intrinsic_sb_to_string(ptr %t2997)
  %t2999 = ptrtoint ptr %t2998 to i64
  ret i64 %t2999
}

define i64 @"format_block"(i64 %block, i64 %indent) {
entry:
  %local.sb.3000 = alloca i64
  %local.stmts.3001 = alloca i64
  %local.result.3002 = alloca i64
  %local.n.3003 = alloca i64
  %local.i.3004 = alloca i64
  %local.stmt.3005 = alloca i64
  %local.block = alloca i64
  store i64 %block, ptr %local.block
  %local.indent = alloca i64
  store i64 %indent, ptr %local.indent
  %t3006 = call ptr @intrinsic_sb_new()
  %t3007 = ptrtoint ptr %t3006 to i64
  store i64 %t3007, ptr %local.sb.3000
  %t3008 = load i64, ptr %local.sb.3000
  %t3009 = call ptr @intrinsic_string_new(ptr @.str.main.156)
  %t3010 = ptrtoint ptr %t3009 to i64
  %t3011 = inttoptr i64 %t3008 to ptr
  %t3012 = inttoptr i64 %t3010 to ptr
  call void @intrinsic_sb_append(ptr %t3011, ptr %t3012)
  %t3013 = load i64, ptr %local.block
  %t3014 = inttoptr i64 %t3013 to ptr
  %t3015 = call ptr @load_ptr(ptr %t3014, i64 1)
  %t3016 = ptrtoint ptr %t3015 to i64
  store i64 %t3016, ptr %local.stmts.3001
  %t3017 = load i64, ptr %local.block
  %t3018 = inttoptr i64 %t3017 to ptr
  %t3019 = call ptr @load_ptr(ptr %t3018, i64 2)
  %t3020 = ptrtoint ptr %t3019 to i64
  store i64 %t3020, ptr %local.result.3002
  %t3021 = load i64, ptr %local.stmts.3001
  %t3022 = inttoptr i64 %t3021 to ptr
  %t3023 = call i64 @intrinsic_vec_len(ptr %t3022)
  store i64 %t3023, ptr %local.n.3003
  store i64 0, ptr %local.i.3004
  br label %loop198
loop198:
  %t3024 = load i64, ptr %local.i.3004
  %t3025 = load i64, ptr %local.n.3003
  %t3026 = icmp slt i64 %t3024, %t3025
  %t3027 = zext i1 %t3026 to i64
  %t3028 = icmp ne i64 %t3027, 0
  br i1 %t3028, label %body198, label %endloop198
body198:
  %t3029 = load i64, ptr %local.stmts.3001
  %t3030 = load i64, ptr %local.i.3004
  %t3031 = inttoptr i64 %t3029 to ptr
  %t3032 = call ptr @intrinsic_vec_get(ptr %t3031, i64 %t3030)
  %t3033 = ptrtoint ptr %t3032 to i64
  store i64 %t3033, ptr %local.stmt.3005
  %t3034 = load i64, ptr %local.sb.3000
  %t3035 = load i64, ptr %local.indent
  %t3036 = add i64 %t3035, 1
  %t3037 = call i64 @"make_indent"(i64 %t3036)
  %t3038 = inttoptr i64 %t3034 to ptr
  %t3039 = inttoptr i64 %t3037 to ptr
  call void @intrinsic_sb_append(ptr %t3038, ptr %t3039)
  %t3040 = load i64, ptr %local.sb.3000
  %t3041 = load i64, ptr %local.stmt.3005
  %t3042 = load i64, ptr %local.indent
  %t3043 = add i64 %t3042, 1
  %t3044 = call i64 @"format_stmt"(i64 %t3041, i64 %t3043)
  %t3045 = inttoptr i64 %t3040 to ptr
  %t3046 = inttoptr i64 %t3044 to ptr
  call void @intrinsic_sb_append(ptr %t3045, ptr %t3046)
  %t3047 = load i64, ptr %local.sb.3000
  %t3048 = call ptr @intrinsic_string_new(ptr @.str.main.157)
  %t3049 = ptrtoint ptr %t3048 to i64
  %t3050 = inttoptr i64 %t3047 to ptr
  %t3051 = inttoptr i64 %t3049 to ptr
  call void @intrinsic_sb_append(ptr %t3050, ptr %t3051)
  %t3052 = load i64, ptr %local.i.3004
  %t3053 = add i64 %t3052, 1
  store i64 %t3053, ptr %local.i.3004
  br label %loop198
endloop198:
  %t3054 = load i64, ptr %local.result.3002
  %t3055 = icmp ne i64 %t3054, 0
  %t3056 = zext i1 %t3055 to i64
  %t3057 = icmp ne i64 %t3056, 0
  br i1 %t3057, label %then199, label %else199
then199:
  %t3058 = load i64, ptr %local.sb.3000
  %t3059 = load i64, ptr %local.indent
  %t3060 = add i64 %t3059, 1
  %t3061 = call i64 @"make_indent"(i64 %t3060)
  %t3062 = inttoptr i64 %t3058 to ptr
  %t3063 = inttoptr i64 %t3061 to ptr
  call void @intrinsic_sb_append(ptr %t3062, ptr %t3063)
  %t3064 = load i64, ptr %local.sb.3000
  %t3065 = load i64, ptr %local.result.3002
  %t3066 = load i64, ptr %local.indent
  %t3067 = add i64 %t3066, 1
  %t3068 = call i64 @"format_expr"(i64 %t3065, i64 %t3067)
  %t3069 = inttoptr i64 %t3064 to ptr
  %t3070 = inttoptr i64 %t3068 to ptr
  call void @intrinsic_sb_append(ptr %t3069, ptr %t3070)
  %t3071 = load i64, ptr %local.sb.3000
  %t3072 = call ptr @intrinsic_string_new(ptr @.str.main.158)
  %t3073 = ptrtoint ptr %t3072 to i64
  %t3074 = inttoptr i64 %t3071 to ptr
  %t3075 = inttoptr i64 %t3073 to ptr
  call void @intrinsic_sb_append(ptr %t3074, ptr %t3075)
  br label %then199_end
then199_end:
  br label %endif199
else199:
  br label %else199_end
else199_end:
  br label %endif199
endif199:
  %t3076 = phi i64 [ 0, %then199_end ], [ 0, %else199_end ]
  %t3077 = load i64, ptr %local.sb.3000
  %t3078 = load i64, ptr %local.indent
  %t3079 = call i64 @"make_indent"(i64 %t3078)
  %t3080 = inttoptr i64 %t3077 to ptr
  %t3081 = inttoptr i64 %t3079 to ptr
  call void @intrinsic_sb_append(ptr %t3080, ptr %t3081)
  %t3082 = load i64, ptr %local.sb.3000
  %t3083 = call ptr @intrinsic_string_new(ptr @.str.main.159)
  %t3084 = ptrtoint ptr %t3083 to i64
  %t3085 = inttoptr i64 %t3082 to ptr
  %t3086 = inttoptr i64 %t3084 to ptr
  call void @intrinsic_sb_append(ptr %t3085, ptr %t3086)
  %t3087 = load i64, ptr %local.sb.3000
  %t3088 = inttoptr i64 %t3087 to ptr
  %t3089 = call ptr @intrinsic_sb_to_string(ptr %t3088)
  %t3090 = ptrtoint ptr %t3089 to i64
  ret i64 %t3090
}

define i64 @"make_indent"(i64 %level) {
entry:
  %local.sb.3091 = alloca i64
  %local.i.3092 = alloca i64
  %local.level = alloca i64
  store i64 %level, ptr %local.level
  %t3093 = call ptr @intrinsic_sb_new()
  %t3094 = ptrtoint ptr %t3093 to i64
  store i64 %t3094, ptr %local.sb.3091
  store i64 0, ptr %local.i.3092
  br label %loop200
loop200:
  %t3095 = load i64, ptr %local.i.3092
  %t3096 = load i64, ptr %local.level
  %t3097 = icmp slt i64 %t3095, %t3096
  %t3098 = zext i1 %t3097 to i64
  %t3099 = icmp ne i64 %t3098, 0
  br i1 %t3099, label %body200, label %endloop200
body200:
  %t3100 = load i64, ptr %local.sb.3091
  %t3101 = call ptr @intrinsic_string_new(ptr @.str.main.160)
  %t3102 = ptrtoint ptr %t3101 to i64
  %t3103 = inttoptr i64 %t3100 to ptr
  %t3104 = inttoptr i64 %t3102 to ptr
  call void @intrinsic_sb_append(ptr %t3103, ptr %t3104)
  %t3105 = load i64, ptr %local.i.3092
  %t3106 = add i64 %t3105, 1
  store i64 %t3106, ptr %local.i.3092
  br label %loop200
endloop200:
  %t3107 = load i64, ptr %local.sb.3091
  %t3108 = inttoptr i64 %t3107 to ptr
  %t3109 = call ptr @intrinsic_sb_to_string(ptr %t3108)
  %t3110 = ptrtoint ptr %t3109 to i64
  ret i64 %t3110
}

define i64 @"format_stmt"(i64 %stmt, i64 %indent) {
entry:
  %local.tag.3111 = alloca i64
  %local.name.3112 = alloca i64
  %local.ty.3113 = alloca i64
  %local.init_expr.3114 = alloca i64
  %local.sb.3115 = alloca i64
  %local.expr.3116 = alloca i64
  %local.formatted.3117 = alloca i64
  %local.expr.3118 = alloca i64
  %local.name.3119 = alloca i64
  %local.value.3120 = alloca i64
  %local.stmt = alloca i64
  store i64 %stmt, ptr %local.stmt
  %local.indent = alloca i64
  store i64 %indent, ptr %local.indent
  %t3121 = load i64, ptr %local.stmt
  %t3122 = inttoptr i64 %t3121 to ptr
  %t3123 = call i64 @load_i64(ptr %t3122, i64 0)
  store i64 %t3123, ptr %local.tag.3111
  %t3124 = load i64, ptr %local.tag.3111
  %t3125 = icmp eq i64 %t3124, 0
  %t3126 = zext i1 %t3125 to i64
  %t3127 = icmp ne i64 %t3126, 0
  br i1 %t3127, label %then201, label %else201
then201:
  %t3128 = load i64, ptr %local.stmt
  %t3129 = inttoptr i64 %t3128 to ptr
  %t3130 = call ptr @load_ptr(ptr %t3129, i64 1)
  %t3131 = ptrtoint ptr %t3130 to i64
  store i64 %t3131, ptr %local.name.3112
  %t3132 = load i64, ptr %local.stmt
  %t3133 = inttoptr i64 %t3132 to ptr
  %t3134 = call ptr @load_ptr(ptr %t3133, i64 2)
  %t3135 = ptrtoint ptr %t3134 to i64
  store i64 %t3135, ptr %local.ty.3113
  %t3136 = load i64, ptr %local.stmt
  %t3137 = inttoptr i64 %t3136 to ptr
  %t3138 = call ptr @load_ptr(ptr %t3137, i64 3)
  %t3139 = ptrtoint ptr %t3138 to i64
  store i64 %t3139, ptr %local.init_expr.3114
  %t3140 = call ptr @intrinsic_sb_new()
  %t3141 = ptrtoint ptr %t3140 to i64
  store i64 %t3141, ptr %local.sb.3115
  %t3142 = load i64, ptr %local.sb.3115
  %t3143 = call ptr @intrinsic_string_new(ptr @.str.main.161)
  %t3144 = ptrtoint ptr %t3143 to i64
  %t3145 = inttoptr i64 %t3142 to ptr
  %t3146 = inttoptr i64 %t3144 to ptr
  call void @intrinsic_sb_append(ptr %t3145, ptr %t3146)
  %t3147 = load i64, ptr %local.sb.3115
  %t3148 = load i64, ptr %local.name.3112
  %t3149 = inttoptr i64 %t3147 to ptr
  %t3150 = inttoptr i64 %t3148 to ptr
  call void @intrinsic_sb_append(ptr %t3149, ptr %t3150)
  %t3151 = load i64, ptr %local.ty.3113
  %t3152 = icmp ne i64 %t3151, 0
  %t3153 = zext i1 %t3152 to i64
  %t3154 = icmp ne i64 %t3153, 0
  br i1 %t3154, label %then202, label %else202
then202:
  %t3155 = load i64, ptr %local.sb.3115
  %t3156 = call ptr @intrinsic_string_new(ptr @.str.main.162)
  %t3157 = ptrtoint ptr %t3156 to i64
  %t3158 = inttoptr i64 %t3155 to ptr
  %t3159 = inttoptr i64 %t3157 to ptr
  call void @intrinsic_sb_append(ptr %t3158, ptr %t3159)
  %t3160 = load i64, ptr %local.sb.3115
  %t3161 = load i64, ptr %local.ty.3113
  %t3162 = inttoptr i64 %t3160 to ptr
  %t3163 = inttoptr i64 %t3161 to ptr
  call void @intrinsic_sb_append(ptr %t3162, ptr %t3163)
  br label %then202_end
then202_end:
  br label %endif202
else202:
  br label %else202_end
else202_end:
  br label %endif202
endif202:
  %t3164 = phi i64 [ 0, %then202_end ], [ 0, %else202_end ]
  %t3165 = load i64, ptr %local.init_expr.3114
  %t3166 = icmp ne i64 %t3165, 0
  %t3167 = zext i1 %t3166 to i64
  %t3168 = icmp ne i64 %t3167, 0
  br i1 %t3168, label %then203, label %else203
then203:
  %t3169 = load i64, ptr %local.sb.3115
  %t3170 = call ptr @intrinsic_string_new(ptr @.str.main.163)
  %t3171 = ptrtoint ptr %t3170 to i64
  %t3172 = inttoptr i64 %t3169 to ptr
  %t3173 = inttoptr i64 %t3171 to ptr
  call void @intrinsic_sb_append(ptr %t3172, ptr %t3173)
  %t3174 = load i64, ptr %local.sb.3115
  %t3175 = load i64, ptr %local.init_expr.3114
  %t3176 = load i64, ptr %local.indent
  %t3177 = call i64 @"format_expr"(i64 %t3175, i64 %t3176)
  %t3178 = inttoptr i64 %t3174 to ptr
  %t3179 = inttoptr i64 %t3177 to ptr
  call void @intrinsic_sb_append(ptr %t3178, ptr %t3179)
  br label %then203_end
then203_end:
  br label %endif203
else203:
  br label %else203_end
else203_end:
  br label %endif203
endif203:
  %t3180 = phi i64 [ 0, %then203_end ], [ 0, %else203_end ]
  %t3181 = load i64, ptr %local.sb.3115
  %t3182 = call ptr @intrinsic_string_new(ptr @.str.main.164)
  %t3183 = ptrtoint ptr %t3182 to i64
  %t3184 = inttoptr i64 %t3181 to ptr
  %t3185 = inttoptr i64 %t3183 to ptr
  call void @intrinsic_sb_append(ptr %t3184, ptr %t3185)
  %t3186 = load i64, ptr %local.sb.3115
  %t3187 = inttoptr i64 %t3186 to ptr
  %t3188 = call ptr @intrinsic_sb_to_string(ptr %t3187)
  %t3189 = ptrtoint ptr %t3188 to i64
  ret i64 %t3189
  br label %then201_end
then201_end:
  br label %endif201
else201:
  br label %else201_end
else201_end:
  br label %endif201
endif201:
  %t3190 = phi i64 [ 0, %then201_end ], [ 0, %else201_end ]
  %t3191 = load i64, ptr %local.tag.3111
  %t3192 = icmp eq i64 %t3191, 1
  %t3193 = zext i1 %t3192 to i64
  %t3194 = icmp ne i64 %t3193, 0
  br i1 %t3194, label %then204, label %else204
then204:
  %t3195 = load i64, ptr %local.stmt
  %t3196 = inttoptr i64 %t3195 to ptr
  %t3197 = call ptr @load_ptr(ptr %t3196, i64 1)
  %t3198 = ptrtoint ptr %t3197 to i64
  store i64 %t3198, ptr %local.expr.3116
  %t3199 = load i64, ptr %local.expr.3116
  %t3200 = load i64, ptr %local.indent
  %t3201 = call i64 @"format_expr"(i64 %t3199, i64 %t3200)
  store i64 %t3201, ptr %local.formatted.3117
  %t3202 = load i64, ptr %local.formatted.3117
  %t3203 = call ptr @intrinsic_string_new(ptr @.str.main.165)
  %t3204 = ptrtoint ptr %t3203 to i64
  %t3205 = inttoptr i64 %t3202 to ptr
  %t3206 = inttoptr i64 %t3204 to ptr
  %t3207 = call ptr @intrinsic_string_concat(ptr %t3205, ptr %t3206)
  %t3208 = ptrtoint ptr %t3207 to i64
  ret i64 %t3208
  br label %then204_end
then204_end:
  br label %endif204
else204:
  br label %else204_end
else204_end:
  br label %endif204
endif204:
  %t3209 = phi i64 [ 0, %then204_end ], [ 0, %else204_end ]
  %t3210 = load i64, ptr %local.tag.3111
  %t3211 = icmp eq i64 %t3210, 2
  %t3212 = zext i1 %t3211 to i64
  %t3213 = icmp ne i64 %t3212, 0
  br i1 %t3213, label %then205, label %else205
then205:
  %t3214 = load i64, ptr %local.stmt
  %t3215 = inttoptr i64 %t3214 to ptr
  %t3216 = call ptr @load_ptr(ptr %t3215, i64 1)
  %t3217 = ptrtoint ptr %t3216 to i64
  store i64 %t3217, ptr %local.expr.3118
  %t3218 = load i64, ptr %local.expr.3118
  %t3219 = icmp eq i64 %t3218, 0
  %t3220 = zext i1 %t3219 to i64
  %t3221 = icmp ne i64 %t3220, 0
  br i1 %t3221, label %then206, label %else206
then206:
  %t3222 = call ptr @intrinsic_string_new(ptr @.str.main.166)
  %t3223 = ptrtoint ptr %t3222 to i64
  ret i64 %t3223
  br label %then206_end
then206_end:
  br label %endif206
else206:
  br label %else206_end
else206_end:
  br label %endif206
endif206:
  %t3224 = phi i64 [ 0, %then206_end ], [ 0, %else206_end ]
  %t3225 = call ptr @intrinsic_string_new(ptr @.str.main.167)
  %t3226 = ptrtoint ptr %t3225 to i64
  %t3227 = load i64, ptr %local.expr.3118
  %t3228 = load i64, ptr %local.indent
  %t3229 = call i64 @"format_expr"(i64 %t3227, i64 %t3228)
  %t3230 = call ptr @intrinsic_string_new(ptr @.str.main.168)
  %t3231 = ptrtoint ptr %t3230 to i64
  %t3232 = inttoptr i64 %t3229 to ptr
  %t3233 = inttoptr i64 %t3231 to ptr
  %t3234 = call ptr @intrinsic_string_concat(ptr %t3232, ptr %t3233)
  %t3235 = ptrtoint ptr %t3234 to i64
  %t3236 = inttoptr i64 %t3226 to ptr
  %t3237 = inttoptr i64 %t3235 to ptr
  %t3238 = call ptr @intrinsic_string_concat(ptr %t3236, ptr %t3237)
  %t3239 = ptrtoint ptr %t3238 to i64
  ret i64 %t3239
  br label %then205_end
then205_end:
  br label %endif205
else205:
  br label %else205_end
else205_end:
  br label %endif205
endif205:
  %t3240 = phi i64 [ 0, %then205_end ], [ 0, %else205_end ]
  %t3241 = load i64, ptr %local.tag.3111
  %t3242 = icmp eq i64 %t3241, 3
  %t3243 = zext i1 %t3242 to i64
  %t3244 = icmp ne i64 %t3243, 0
  br i1 %t3244, label %then207, label %else207
then207:
  %t3245 = load i64, ptr %local.stmt
  %t3246 = inttoptr i64 %t3245 to ptr
  %t3247 = call ptr @load_ptr(ptr %t3246, i64 1)
  %t3248 = ptrtoint ptr %t3247 to i64
  store i64 %t3248, ptr %local.name.3119
  %t3249 = load i64, ptr %local.stmt
  %t3250 = inttoptr i64 %t3249 to ptr
  %t3251 = call ptr @load_ptr(ptr %t3250, i64 2)
  %t3252 = ptrtoint ptr %t3251 to i64
  store i64 %t3252, ptr %local.value.3120
  %t3253 = load i64, ptr %local.name.3119
  %t3254 = call ptr @intrinsic_string_new(ptr @.str.main.169)
  %t3255 = ptrtoint ptr %t3254 to i64
  %t3256 = load i64, ptr %local.value.3120
  %t3257 = load i64, ptr %local.indent
  %t3258 = call i64 @"format_expr"(i64 %t3256, i64 %t3257)
  %t3259 = call ptr @intrinsic_string_new(ptr @.str.main.170)
  %t3260 = ptrtoint ptr %t3259 to i64
  %t3261 = inttoptr i64 %t3258 to ptr
  %t3262 = inttoptr i64 %t3260 to ptr
  %t3263 = call ptr @intrinsic_string_concat(ptr %t3261, ptr %t3262)
  %t3264 = ptrtoint ptr %t3263 to i64
  %t3265 = inttoptr i64 %t3255 to ptr
  %t3266 = inttoptr i64 %t3264 to ptr
  %t3267 = call ptr @intrinsic_string_concat(ptr %t3265, ptr %t3266)
  %t3268 = ptrtoint ptr %t3267 to i64
  %t3269 = inttoptr i64 %t3253 to ptr
  %t3270 = inttoptr i64 %t3268 to ptr
  %t3271 = call ptr @intrinsic_string_concat(ptr %t3269, ptr %t3270)
  %t3272 = ptrtoint ptr %t3271 to i64
  ret i64 %t3272
  br label %then207_end
then207_end:
  br label %endif207
else207:
  br label %else207_end
else207_end:
  br label %endif207
endif207:
  %t3273 = phi i64 [ 0, %then207_end ], [ 0, %else207_end ]
  %t3274 = load i64, ptr %local.tag.3111
  %t3275 = icmp eq i64 %t3274, 4
  %t3276 = zext i1 %t3275 to i64
  %t3277 = icmp ne i64 %t3276, 0
  br i1 %t3277, label %then208, label %else208
then208:
  %t3278 = call ptr @intrinsic_string_new(ptr @.str.main.171)
  %t3279 = ptrtoint ptr %t3278 to i64
  ret i64 %t3279
  br label %then208_end
then208_end:
  br label %endif208
else208:
  br label %else208_end
else208_end:
  br label %endif208
endif208:
  %t3280 = phi i64 [ 0, %then208_end ], [ 0, %else208_end ]
  %t3281 = load i64, ptr %local.tag.3111
  %t3282 = icmp eq i64 %t3281, 5
  %t3283 = zext i1 %t3282 to i64
  %t3284 = icmp ne i64 %t3283, 0
  br i1 %t3284, label %then209, label %else209
then209:
  %t3285 = call ptr @intrinsic_string_new(ptr @.str.main.172)
  %t3286 = ptrtoint ptr %t3285 to i64
  ret i64 %t3286
  br label %then209_end
then209_end:
  br label %endif209
else209:
  br label %else209_end
else209_end:
  br label %endif209
endif209:
  %t3287 = phi i64 [ 0, %then209_end ], [ 0, %else209_end ]
  %t3288 = call ptr @intrinsic_string_new(ptr @.str.main.173)
  %t3289 = ptrtoint ptr %t3288 to i64
  ret i64 %t3289
}

define i64 @"format_expr"(i64 %expr, i64 %indent) {
entry:
  %local.tag.3290 = alloca i64
  %local.value.3291 = alloca i64
  %local.value.3292 = alloca i64
  %local.value.3293 = alloca i64
  %local.segments.3294 = alloca i64
  %local.sb.3295 = alloca i64
  %local.i.3296 = alloca i64
  %local.callee.3297 = alloca i64
  %local.args.3298 = alloca i64
  %local.sb.3299 = alloca i64
  %local.i.3300 = alloca i64
  %local.op.3301 = alloca i64
  %local.left.3302 = alloca i64
  %local.right.3303 = alloca i64
  %local.op.3304 = alloca i64
  %local.operand.3305 = alloca i64
  %local.cond.3306 = alloca i64
  %local.then_block.3307 = alloca i64
  %local.else_block.3308 = alloca i64
  %local.sb.3309 = alloca i64
  %local.else_tag.3310 = alloca i64
  %local.cond.3311 = alloca i64
  %local.body.3312 = alloca i64
  %local.name.3313 = alloca i64
  %local.fields.3314 = alloca i64
  %local.sb.3315 = alloca i64
  %local.i.3316 = alloca i64
  %local.field.3317 = alloca i64
  %local.fname.3318 = alloca i64
  %local.fval.3319 = alloca i64
  %local.obj.3320 = alloca i64
  %local.field.3321 = alloca i64
  %local.loop_var.3322 = alloca i64
  %local.start.3323 = alloca i64
  %local.end.3324 = alloca i64
  %local.body.3325 = alloca i64
  %local.scrutinee.3326 = alloca i64
  %local.arms.3327 = alloca i64
  %local.sb.3328 = alloca i64
  %local.i.3329 = alloca i64
  %local.arm.3330 = alloca i64
  %local.pattern.3331 = alloca i64
  %local.result.3332 = alloca i64
  %local.obj.3333 = alloca i64
  %local.method.3334 = alloca i64
  %local.args.3335 = alloca i64
  %local.sb.3336 = alloca i64
  %local.i.3337 = alloca i64
  %local.inner.3338 = alloca i64
  %local.expr = alloca i64
  store i64 %expr, ptr %local.expr
  %local.indent = alloca i64
  store i64 %indent, ptr %local.indent
  %t3339 = load i64, ptr %local.expr
  %t3340 = icmp eq i64 %t3339, 0
  %t3341 = zext i1 %t3340 to i64
  %t3342 = icmp ne i64 %t3341, 0
  br i1 %t3342, label %then210, label %else210
then210:
  %t3343 = call ptr @intrinsic_string_new(ptr @.str.main.174)
  %t3344 = ptrtoint ptr %t3343 to i64
  ret i64 %t3344
  br label %then210_end
then210_end:
  br label %endif210
else210:
  br label %else210_end
else210_end:
  br label %endif210
endif210:
  %t3345 = phi i64 [ 0, %then210_end ], [ 0, %else210_end ]
  %t3346 = load i64, ptr %local.expr
  %t3347 = inttoptr i64 %t3346 to ptr
  %t3348 = call i64 @load_i64(ptr %t3347, i64 0)
  store i64 %t3348, ptr %local.tag.3290
  %t3349 = load i64, ptr %local.tag.3290
  %t3350 = icmp eq i64 %t3349, 0
  %t3351 = zext i1 %t3350 to i64
  %t3352 = icmp ne i64 %t3351, 0
  br i1 %t3352, label %then211, label %else211
then211:
  %t3353 = load i64, ptr %local.expr
  %t3354 = inttoptr i64 %t3353 to ptr
  %t3355 = call i64 @load_i64(ptr %t3354, i64 1)
  store i64 %t3355, ptr %local.value.3291
  %t3356 = load i64, ptr %local.value.3291
  %t3357 = call ptr @intrinsic_int_to_string(i64 %t3356)
  %t3358 = ptrtoint ptr %t3357 to i64
  ret i64 %t3358
  br label %then211_end
then211_end:
  br label %endif211
else211:
  br label %else211_end
else211_end:
  br label %endif211
endif211:
  %t3359 = phi i64 [ 0, %then211_end ], [ 0, %else211_end ]
  %t3360 = load i64, ptr %local.tag.3290
  %t3361 = icmp eq i64 %t3360, 1
  %t3362 = zext i1 %t3361 to i64
  %t3363 = icmp ne i64 %t3362, 0
  br i1 %t3363, label %then212, label %else212
then212:
  %t3364 = load i64, ptr %local.expr
  %t3365 = inttoptr i64 %t3364 to ptr
  %t3366 = call i64 @load_i64(ptr %t3365, i64 1)
  store i64 %t3366, ptr %local.value.3292
  %t3367 = load i64, ptr %local.value.3292
  %t3368 = icmp ne i64 %t3367, 0
  %t3369 = zext i1 %t3368 to i64
  %t3370 = icmp ne i64 %t3369, 0
  br i1 %t3370, label %then213, label %else213
then213:
  %t3371 = call ptr @intrinsic_string_new(ptr @.str.main.175)
  %t3372 = ptrtoint ptr %t3371 to i64
  ret i64 %t3372
  br label %then213_end
then213_end:
  br label %endif213
else213:
  br label %else213_end
else213_end:
  br label %endif213
endif213:
  %t3373 = phi i64 [ 0, %then213_end ], [ 0, %else213_end ]
  %t3374 = call ptr @intrinsic_string_new(ptr @.str.main.176)
  %t3375 = ptrtoint ptr %t3374 to i64
  ret i64 %t3375
  br label %then212_end
then212_end:
  br label %endif212
else212:
  br label %else212_end
else212_end:
  br label %endif212
endif212:
  %t3376 = phi i64 [ 0, %then212_end ], [ 0, %else212_end ]
  %t3377 = load i64, ptr %local.tag.3290
  %t3378 = icmp eq i64 %t3377, 2
  %t3379 = zext i1 %t3378 to i64
  %t3380 = icmp ne i64 %t3379, 0
  br i1 %t3380, label %then214, label %else214
then214:
  %t3381 = load i64, ptr %local.expr
  %t3382 = inttoptr i64 %t3381 to ptr
  %t3383 = call ptr @load_ptr(ptr %t3382, i64 1)
  %t3384 = ptrtoint ptr %t3383 to i64
  store i64 %t3384, ptr %local.value.3293
  %t3385 = call ptr @intrinsic_string_new(ptr @.str.main.177)
  %t3386 = ptrtoint ptr %t3385 to i64
  %t3387 = load i64, ptr %local.value.3293
  %t3388 = call ptr @intrinsic_string_new(ptr @.str.main.178)
  %t3389 = ptrtoint ptr %t3388 to i64
  %t3390 = inttoptr i64 %t3387 to ptr
  %t3391 = inttoptr i64 %t3389 to ptr
  %t3392 = call ptr @intrinsic_string_concat(ptr %t3390, ptr %t3391)
  %t3393 = ptrtoint ptr %t3392 to i64
  %t3394 = inttoptr i64 %t3386 to ptr
  %t3395 = inttoptr i64 %t3393 to ptr
  %t3396 = call ptr @intrinsic_string_concat(ptr %t3394, ptr %t3395)
  %t3397 = ptrtoint ptr %t3396 to i64
  ret i64 %t3397
  br label %then214_end
then214_end:
  br label %endif214
else214:
  br label %else214_end
else214_end:
  br label %endif214
endif214:
  %t3398 = phi i64 [ 0, %then214_end ], [ 0, %else214_end ]
  %t3399 = load i64, ptr %local.tag.3290
  %t3400 = icmp eq i64 %t3399, 3
  %t3401 = zext i1 %t3400 to i64
  %t3402 = icmp ne i64 %t3401, 0
  br i1 %t3402, label %then215, label %else215
then215:
  %t3403 = load i64, ptr %local.expr
  %t3404 = inttoptr i64 %t3403 to ptr
  %t3405 = call ptr @load_ptr(ptr %t3404, i64 1)
  %t3406 = ptrtoint ptr %t3405 to i64
  ret i64 %t3406
  br label %then215_end
then215_end:
  br label %endif215
else215:
  br label %else215_end
else215_end:
  br label %endif215
endif215:
  %t3407 = phi i64 [ 0, %then215_end ], [ 0, %else215_end ]
  %t3408 = load i64, ptr %local.tag.3290
  %t3409 = icmp eq i64 %t3408, 4
  %t3410 = zext i1 %t3409 to i64
  %t3411 = icmp ne i64 %t3410, 0
  br i1 %t3411, label %then216, label %else216
then216:
  %t3412 = load i64, ptr %local.expr
  %t3413 = inttoptr i64 %t3412 to ptr
  %t3414 = call ptr @load_ptr(ptr %t3413, i64 1)
  %t3415 = ptrtoint ptr %t3414 to i64
  store i64 %t3415, ptr %local.segments.3294
  %t3416 = call ptr @intrinsic_sb_new()
  %t3417 = ptrtoint ptr %t3416 to i64
  store i64 %t3417, ptr %local.sb.3295
  store i64 0, ptr %local.i.3296
  br label %loop217
loop217:
  %t3418 = load i64, ptr %local.i.3296
  %t3419 = load i64, ptr %local.segments.3294
  %t3420 = inttoptr i64 %t3419 to ptr
  %t3421 = call i64 @intrinsic_vec_len(ptr %t3420)
  %t3422 = icmp slt i64 %t3418, %t3421
  %t3423 = zext i1 %t3422 to i64
  %t3424 = icmp ne i64 %t3423, 0
  br i1 %t3424, label %body217, label %endloop217
body217:
  %t3425 = load i64, ptr %local.i.3296
  %t3426 = icmp sgt i64 %t3425, 0
  %t3427 = zext i1 %t3426 to i64
  %t3428 = icmp ne i64 %t3427, 0
  br i1 %t3428, label %then218, label %else218
then218:
  %t3429 = load i64, ptr %local.sb.3295
  %t3430 = call ptr @intrinsic_string_new(ptr @.str.main.179)
  %t3431 = ptrtoint ptr %t3430 to i64
  %t3432 = inttoptr i64 %t3429 to ptr
  %t3433 = inttoptr i64 %t3431 to ptr
  call void @intrinsic_sb_append(ptr %t3432, ptr %t3433)
  br label %then218_end
then218_end:
  br label %endif218
else218:
  br label %else218_end
else218_end:
  br label %endif218
endif218:
  %t3434 = phi i64 [ 0, %then218_end ], [ 0, %else218_end ]
  %t3435 = load i64, ptr %local.sb.3295
  %t3436 = load i64, ptr %local.segments.3294
  %t3437 = load i64, ptr %local.i.3296
  %t3438 = inttoptr i64 %t3436 to ptr
  %t3439 = call ptr @intrinsic_vec_get(ptr %t3438, i64 %t3437)
  %t3440 = ptrtoint ptr %t3439 to i64
  %t3441 = inttoptr i64 %t3435 to ptr
  %t3442 = inttoptr i64 %t3440 to ptr
  call void @intrinsic_sb_append(ptr %t3441, ptr %t3442)
  %t3443 = load i64, ptr %local.i.3296
  %t3444 = add i64 %t3443, 1
  store i64 %t3444, ptr %local.i.3296
  br label %loop217
endloop217:
  %t3445 = load i64, ptr %local.sb.3295
  %t3446 = inttoptr i64 %t3445 to ptr
  %t3447 = call ptr @intrinsic_sb_to_string(ptr %t3446)
  %t3448 = ptrtoint ptr %t3447 to i64
  ret i64 %t3448
  br label %then216_end
then216_end:
  br label %endif216
else216:
  br label %else216_end
else216_end:
  br label %endif216
endif216:
  %t3449 = phi i64 [ 0, %then216_end ], [ 0, %else216_end ]
  %t3450 = load i64, ptr %local.tag.3290
  %t3451 = icmp eq i64 %t3450, 5
  %t3452 = zext i1 %t3451 to i64
  %t3453 = icmp ne i64 %t3452, 0
  br i1 %t3453, label %then219, label %else219
then219:
  %t3454 = load i64, ptr %local.expr
  %t3455 = inttoptr i64 %t3454 to ptr
  %t3456 = call ptr @load_ptr(ptr %t3455, i64 1)
  %t3457 = ptrtoint ptr %t3456 to i64
  store i64 %t3457, ptr %local.callee.3297
  %t3458 = load i64, ptr %local.expr
  %t3459 = inttoptr i64 %t3458 to ptr
  %t3460 = call ptr @load_ptr(ptr %t3459, i64 2)
  %t3461 = ptrtoint ptr %t3460 to i64
  store i64 %t3461, ptr %local.args.3298
  %t3462 = call ptr @intrinsic_sb_new()
  %t3463 = ptrtoint ptr %t3462 to i64
  store i64 %t3463, ptr %local.sb.3299
  %t3464 = load i64, ptr %local.sb.3299
  %t3465 = load i64, ptr %local.callee.3297
  %t3466 = load i64, ptr %local.indent
  %t3467 = call i64 @"format_expr"(i64 %t3465, i64 %t3466)
  %t3468 = inttoptr i64 %t3464 to ptr
  %t3469 = inttoptr i64 %t3467 to ptr
  call void @intrinsic_sb_append(ptr %t3468, ptr %t3469)
  %t3470 = load i64, ptr %local.sb.3299
  %t3471 = call ptr @intrinsic_string_new(ptr @.str.main.180)
  %t3472 = ptrtoint ptr %t3471 to i64
  %t3473 = inttoptr i64 %t3470 to ptr
  %t3474 = inttoptr i64 %t3472 to ptr
  call void @intrinsic_sb_append(ptr %t3473, ptr %t3474)
  store i64 0, ptr %local.i.3300
  br label %loop220
loop220:
  %t3475 = load i64, ptr %local.i.3300
  %t3476 = load i64, ptr %local.args.3298
  %t3477 = inttoptr i64 %t3476 to ptr
  %t3478 = call i64 @intrinsic_vec_len(ptr %t3477)
  %t3479 = icmp slt i64 %t3475, %t3478
  %t3480 = zext i1 %t3479 to i64
  %t3481 = icmp ne i64 %t3480, 0
  br i1 %t3481, label %body220, label %endloop220
body220:
  %t3482 = load i64, ptr %local.i.3300
  %t3483 = icmp sgt i64 %t3482, 0
  %t3484 = zext i1 %t3483 to i64
  %t3485 = icmp ne i64 %t3484, 0
  br i1 %t3485, label %then221, label %else221
then221:
  %t3486 = load i64, ptr %local.sb.3299
  %t3487 = call ptr @intrinsic_string_new(ptr @.str.main.181)
  %t3488 = ptrtoint ptr %t3487 to i64
  %t3489 = inttoptr i64 %t3486 to ptr
  %t3490 = inttoptr i64 %t3488 to ptr
  call void @intrinsic_sb_append(ptr %t3489, ptr %t3490)
  br label %then221_end
then221_end:
  br label %endif221
else221:
  br label %else221_end
else221_end:
  br label %endif221
endif221:
  %t3491 = phi i64 [ 0, %then221_end ], [ 0, %else221_end ]
  %t3492 = load i64, ptr %local.sb.3299
  %t3493 = load i64, ptr %local.args.3298
  %t3494 = load i64, ptr %local.i.3300
  %t3495 = inttoptr i64 %t3493 to ptr
  %t3496 = call ptr @intrinsic_vec_get(ptr %t3495, i64 %t3494)
  %t3497 = ptrtoint ptr %t3496 to i64
  %t3498 = load i64, ptr %local.indent
  %t3499 = call i64 @"format_expr"(i64 %t3497, i64 %t3498)
  %t3500 = inttoptr i64 %t3492 to ptr
  %t3501 = inttoptr i64 %t3499 to ptr
  call void @intrinsic_sb_append(ptr %t3500, ptr %t3501)
  %t3502 = load i64, ptr %local.i.3300
  %t3503 = add i64 %t3502, 1
  store i64 %t3503, ptr %local.i.3300
  br label %loop220
endloop220:
  %t3504 = load i64, ptr %local.sb.3299
  %t3505 = call ptr @intrinsic_string_new(ptr @.str.main.182)
  %t3506 = ptrtoint ptr %t3505 to i64
  %t3507 = inttoptr i64 %t3504 to ptr
  %t3508 = inttoptr i64 %t3506 to ptr
  call void @intrinsic_sb_append(ptr %t3507, ptr %t3508)
  %t3509 = load i64, ptr %local.sb.3299
  %t3510 = inttoptr i64 %t3509 to ptr
  %t3511 = call ptr @intrinsic_sb_to_string(ptr %t3510)
  %t3512 = ptrtoint ptr %t3511 to i64
  ret i64 %t3512
  br label %then219_end
then219_end:
  br label %endif219
else219:
  br label %else219_end
else219_end:
  br label %endif219
endif219:
  %t3513 = phi i64 [ 0, %then219_end ], [ 0, %else219_end ]
  %t3514 = load i64, ptr %local.tag.3290
  %t3515 = icmp eq i64 %t3514, 6
  %t3516 = zext i1 %t3515 to i64
  %t3517 = icmp ne i64 %t3516, 0
  br i1 %t3517, label %then222, label %else222
then222:
  %t3518 = load i64, ptr %local.expr
  %t3519 = inttoptr i64 %t3518 to ptr
  %t3520 = call i64 @load_i64(ptr %t3519, i64 1)
  store i64 %t3520, ptr %local.op.3301
  %t3521 = load i64, ptr %local.expr
  %t3522 = inttoptr i64 %t3521 to ptr
  %t3523 = call ptr @load_ptr(ptr %t3522, i64 2)
  %t3524 = ptrtoint ptr %t3523 to i64
  store i64 %t3524, ptr %local.left.3302
  %t3525 = load i64, ptr %local.expr
  %t3526 = inttoptr i64 %t3525 to ptr
  %t3527 = call ptr @load_ptr(ptr %t3526, i64 3)
  %t3528 = ptrtoint ptr %t3527 to i64
  store i64 %t3528, ptr %local.right.3303
  %t3529 = load i64, ptr %local.left.3302
  %t3530 = load i64, ptr %local.indent
  %t3531 = call i64 @"format_expr"(i64 %t3529, i64 %t3530)
  %t3532 = call ptr @intrinsic_string_new(ptr @.str.main.183)
  %t3533 = ptrtoint ptr %t3532 to i64
  %t3534 = load i64, ptr %local.op.3301
  %t3535 = call i64 @"format_binop"(i64 %t3534)
  %t3536 = call ptr @intrinsic_string_new(ptr @.str.main.184)
  %t3537 = ptrtoint ptr %t3536 to i64
  %t3538 = load i64, ptr %local.right.3303
  %t3539 = load i64, ptr %local.indent
  %t3540 = call i64 @"format_expr"(i64 %t3538, i64 %t3539)
  %t3541 = inttoptr i64 %t3537 to ptr
  %t3542 = inttoptr i64 %t3540 to ptr
  %t3543 = call ptr @intrinsic_string_concat(ptr %t3541, ptr %t3542)
  %t3544 = ptrtoint ptr %t3543 to i64
  %t3545 = inttoptr i64 %t3535 to ptr
  %t3546 = inttoptr i64 %t3544 to ptr
  %t3547 = call ptr @intrinsic_string_concat(ptr %t3545, ptr %t3546)
  %t3548 = ptrtoint ptr %t3547 to i64
  %t3549 = inttoptr i64 %t3533 to ptr
  %t3550 = inttoptr i64 %t3548 to ptr
  %t3551 = call ptr @intrinsic_string_concat(ptr %t3549, ptr %t3550)
  %t3552 = ptrtoint ptr %t3551 to i64
  %t3553 = inttoptr i64 %t3531 to ptr
  %t3554 = inttoptr i64 %t3552 to ptr
  %t3555 = call ptr @intrinsic_string_concat(ptr %t3553, ptr %t3554)
  %t3556 = ptrtoint ptr %t3555 to i64
  ret i64 %t3556
  br label %then222_end
then222_end:
  br label %endif222
else222:
  br label %else222_end
else222_end:
  br label %endif222
endif222:
  %t3557 = phi i64 [ 0, %then222_end ], [ 0, %else222_end ]
  %t3558 = load i64, ptr %local.tag.3290
  %t3559 = icmp eq i64 %t3558, 7
  %t3560 = zext i1 %t3559 to i64
  %t3561 = icmp ne i64 %t3560, 0
  br i1 %t3561, label %then223, label %else223
then223:
  %t3562 = load i64, ptr %local.expr
  %t3563 = inttoptr i64 %t3562 to ptr
  %t3564 = call i64 @load_i64(ptr %t3563, i64 1)
  store i64 %t3564, ptr %local.op.3304
  %t3565 = load i64, ptr %local.expr
  %t3566 = inttoptr i64 %t3565 to ptr
  %t3567 = call ptr @load_ptr(ptr %t3566, i64 2)
  %t3568 = ptrtoint ptr %t3567 to i64
  store i64 %t3568, ptr %local.operand.3305
  %t3569 = load i64, ptr %local.op.3304
  %t3570 = icmp eq i64 %t3569, 100
  %t3571 = zext i1 %t3570 to i64
  %t3572 = icmp ne i64 %t3571, 0
  br i1 %t3572, label %then224, label %else224
then224:
  %t3573 = call ptr @intrinsic_string_new(ptr @.str.main.185)
  %t3574 = ptrtoint ptr %t3573 to i64
  %t3575 = load i64, ptr %local.operand.3305
  %t3576 = load i64, ptr %local.indent
  %t3577 = call i64 @"format_expr"(i64 %t3575, i64 %t3576)
  %t3578 = inttoptr i64 %t3574 to ptr
  %t3579 = inttoptr i64 %t3577 to ptr
  %t3580 = call ptr @intrinsic_string_concat(ptr %t3578, ptr %t3579)
  %t3581 = ptrtoint ptr %t3580 to i64
  ret i64 %t3581
  br label %then224_end
then224_end:
  br label %endif224
else224:
  br label %else224_end
else224_end:
  br label %endif224
endif224:
  %t3582 = phi i64 [ 0, %then224_end ], [ 0, %else224_end ]
  %t3583 = call ptr @intrinsic_string_new(ptr @.str.main.186)
  %t3584 = ptrtoint ptr %t3583 to i64
  %t3585 = load i64, ptr %local.operand.3305
  %t3586 = load i64, ptr %local.indent
  %t3587 = call i64 @"format_expr"(i64 %t3585, i64 %t3586)
  %t3588 = inttoptr i64 %t3584 to ptr
  %t3589 = inttoptr i64 %t3587 to ptr
  %t3590 = call ptr @intrinsic_string_concat(ptr %t3588, ptr %t3589)
  %t3591 = ptrtoint ptr %t3590 to i64
  ret i64 %t3591
  br label %then223_end
then223_end:
  br label %endif223
else223:
  br label %else223_end
else223_end:
  br label %endif223
endif223:
  %t3592 = phi i64 [ 0, %then223_end ], [ 0, %else223_end ]
  %t3593 = load i64, ptr %local.tag.3290
  %t3594 = icmp eq i64 %t3593, 8
  %t3595 = zext i1 %t3594 to i64
  %t3596 = icmp ne i64 %t3595, 0
  br i1 %t3596, label %then225, label %else225
then225:
  %t3597 = load i64, ptr %local.expr
  %t3598 = inttoptr i64 %t3597 to ptr
  %t3599 = call ptr @load_ptr(ptr %t3598, i64 1)
  %t3600 = ptrtoint ptr %t3599 to i64
  store i64 %t3600, ptr %local.cond.3306
  %t3601 = load i64, ptr %local.expr
  %t3602 = inttoptr i64 %t3601 to ptr
  %t3603 = call ptr @load_ptr(ptr %t3602, i64 2)
  %t3604 = ptrtoint ptr %t3603 to i64
  store i64 %t3604, ptr %local.then_block.3307
  %t3605 = load i64, ptr %local.expr
  %t3606 = inttoptr i64 %t3605 to ptr
  %t3607 = call ptr @load_ptr(ptr %t3606, i64 3)
  %t3608 = ptrtoint ptr %t3607 to i64
  store i64 %t3608, ptr %local.else_block.3308
  %t3609 = call ptr @intrinsic_sb_new()
  %t3610 = ptrtoint ptr %t3609 to i64
  store i64 %t3610, ptr %local.sb.3309
  %t3611 = load i64, ptr %local.sb.3309
  %t3612 = call ptr @intrinsic_string_new(ptr @.str.main.187)
  %t3613 = ptrtoint ptr %t3612 to i64
  %t3614 = inttoptr i64 %t3611 to ptr
  %t3615 = inttoptr i64 %t3613 to ptr
  call void @intrinsic_sb_append(ptr %t3614, ptr %t3615)
  %t3616 = load i64, ptr %local.sb.3309
  %t3617 = load i64, ptr %local.cond.3306
  %t3618 = load i64, ptr %local.indent
  %t3619 = call i64 @"format_expr"(i64 %t3617, i64 %t3618)
  %t3620 = inttoptr i64 %t3616 to ptr
  %t3621 = inttoptr i64 %t3619 to ptr
  call void @intrinsic_sb_append(ptr %t3620, ptr %t3621)
  %t3622 = load i64, ptr %local.sb.3309
  %t3623 = call ptr @intrinsic_string_new(ptr @.str.main.188)
  %t3624 = ptrtoint ptr %t3623 to i64
  %t3625 = inttoptr i64 %t3622 to ptr
  %t3626 = inttoptr i64 %t3624 to ptr
  call void @intrinsic_sb_append(ptr %t3625, ptr %t3626)
  %t3627 = load i64, ptr %local.sb.3309
  %t3628 = load i64, ptr %local.then_block.3307
  %t3629 = load i64, ptr %local.indent
  %t3630 = call i64 @"format_block"(i64 %t3628, i64 %t3629)
  %t3631 = inttoptr i64 %t3627 to ptr
  %t3632 = inttoptr i64 %t3630 to ptr
  call void @intrinsic_sb_append(ptr %t3631, ptr %t3632)
  %t3633 = load i64, ptr %local.else_block.3308
  %t3634 = icmp ne i64 %t3633, 0
  %t3635 = zext i1 %t3634 to i64
  %t3636 = icmp ne i64 %t3635, 0
  br i1 %t3636, label %then226, label %else226
then226:
  %t3637 = load i64, ptr %local.sb.3309
  %t3638 = call ptr @intrinsic_string_new(ptr @.str.main.189)
  %t3639 = ptrtoint ptr %t3638 to i64
  %t3640 = inttoptr i64 %t3637 to ptr
  %t3641 = inttoptr i64 %t3639 to ptr
  call void @intrinsic_sb_append(ptr %t3640, ptr %t3641)
  %t3642 = load i64, ptr %local.else_block.3308
  %t3643 = inttoptr i64 %t3642 to ptr
  %t3644 = call i64 @load_i64(ptr %t3643, i64 0)
  store i64 %t3644, ptr %local.else_tag.3310
  %t3645 = load i64, ptr %local.else_tag.3310
  %t3646 = icmp eq i64 %t3645, 8
  %t3647 = zext i1 %t3646 to i64
  %t3648 = icmp ne i64 %t3647, 0
  br i1 %t3648, label %then227, label %else227
then227:
  %t3649 = load i64, ptr %local.sb.3309
  %t3650 = load i64, ptr %local.else_block.3308
  %t3651 = load i64, ptr %local.indent
  %t3652 = call i64 @"format_expr"(i64 %t3650, i64 %t3651)
  %t3653 = inttoptr i64 %t3649 to ptr
  %t3654 = inttoptr i64 %t3652 to ptr
  call void @intrinsic_sb_append(ptr %t3653, ptr %t3654)
  br label %then227_end
then227_end:
  br label %endif227
else227:
  %t3655 = load i64, ptr %local.sb.3309
  %t3656 = load i64, ptr %local.else_block.3308
  %t3657 = load i64, ptr %local.indent
  %t3658 = call i64 @"format_block"(i64 %t3656, i64 %t3657)
  %t3659 = inttoptr i64 %t3655 to ptr
  %t3660 = inttoptr i64 %t3658 to ptr
  call void @intrinsic_sb_append(ptr %t3659, ptr %t3660)
  br label %else227_end
else227_end:
  br label %endif227
endif227:
  %t3661 = phi i64 [ 0, %then227_end ], [ 0, %else227_end ]
  br label %then226_end
then226_end:
  br label %endif226
else226:
  br label %else226_end
else226_end:
  br label %endif226
endif226:
  %t3662 = phi i64 [ %t3661, %then226_end ], [ 0, %else226_end ]
  %t3663 = load i64, ptr %local.sb.3309
  %t3664 = inttoptr i64 %t3663 to ptr
  %t3665 = call ptr @intrinsic_sb_to_string(ptr %t3664)
  %t3666 = ptrtoint ptr %t3665 to i64
  ret i64 %t3666
  br label %then225_end
then225_end:
  br label %endif225
else225:
  br label %else225_end
else225_end:
  br label %endif225
endif225:
  %t3667 = phi i64 [ 0, %then225_end ], [ 0, %else225_end ]
  %t3668 = load i64, ptr %local.tag.3290
  %t3669 = icmp eq i64 %t3668, 9
  %t3670 = zext i1 %t3669 to i64
  %t3671 = icmp ne i64 %t3670, 0
  br i1 %t3671, label %then228, label %else228
then228:
  %t3672 = load i64, ptr %local.expr
  %t3673 = inttoptr i64 %t3672 to ptr
  %t3674 = call ptr @load_ptr(ptr %t3673, i64 1)
  %t3675 = ptrtoint ptr %t3674 to i64
  store i64 %t3675, ptr %local.cond.3311
  %t3676 = load i64, ptr %local.expr
  %t3677 = inttoptr i64 %t3676 to ptr
  %t3678 = call ptr @load_ptr(ptr %t3677, i64 2)
  %t3679 = ptrtoint ptr %t3678 to i64
  store i64 %t3679, ptr %local.body.3312
  %t3680 = call ptr @intrinsic_string_new(ptr @.str.main.190)
  %t3681 = ptrtoint ptr %t3680 to i64
  %t3682 = load i64, ptr %local.cond.3311
  %t3683 = load i64, ptr %local.indent
  %t3684 = call i64 @"format_expr"(i64 %t3682, i64 %t3683)
  %t3685 = call ptr @intrinsic_string_new(ptr @.str.main.191)
  %t3686 = ptrtoint ptr %t3685 to i64
  %t3687 = load i64, ptr %local.body.3312
  %t3688 = load i64, ptr %local.indent
  %t3689 = call i64 @"format_block"(i64 %t3687, i64 %t3688)
  %t3690 = inttoptr i64 %t3686 to ptr
  %t3691 = inttoptr i64 %t3689 to ptr
  %t3692 = call ptr @intrinsic_string_concat(ptr %t3690, ptr %t3691)
  %t3693 = ptrtoint ptr %t3692 to i64
  %t3694 = inttoptr i64 %t3684 to ptr
  %t3695 = inttoptr i64 %t3693 to ptr
  %t3696 = call ptr @intrinsic_string_concat(ptr %t3694, ptr %t3695)
  %t3697 = ptrtoint ptr %t3696 to i64
  %t3698 = inttoptr i64 %t3681 to ptr
  %t3699 = inttoptr i64 %t3697 to ptr
  %t3700 = call ptr @intrinsic_string_concat(ptr %t3698, ptr %t3699)
  %t3701 = ptrtoint ptr %t3700 to i64
  ret i64 %t3701
  br label %then228_end
then228_end:
  br label %endif228
else228:
  br label %else228_end
else228_end:
  br label %endif228
endif228:
  %t3702 = phi i64 [ 0, %then228_end ], [ 0, %else228_end ]
  %t3703 = load i64, ptr %local.tag.3290
  %t3704 = icmp eq i64 %t3703, 10
  %t3705 = zext i1 %t3704 to i64
  %t3706 = icmp ne i64 %t3705, 0
  br i1 %t3706, label %then229, label %else229
then229:
  %t3707 = load i64, ptr %local.expr
  %t3708 = load i64, ptr %local.indent
  %t3709 = call i64 @"format_block"(i64 %t3707, i64 %t3708)
  ret i64 %t3709
  br label %then229_end
then229_end:
  br label %endif229
else229:
  br label %else229_end
else229_end:
  br label %endif229
endif229:
  %t3710 = phi i64 [ 0, %then229_end ], [ 0, %else229_end ]
  %t3711 = load i64, ptr %local.tag.3290
  %t3712 = icmp eq i64 %t3711, 11
  %t3713 = zext i1 %t3712 to i64
  %t3714 = icmp ne i64 %t3713, 0
  br i1 %t3714, label %then230, label %else230
then230:
  %t3715 = load i64, ptr %local.expr
  %t3716 = inttoptr i64 %t3715 to ptr
  %t3717 = call ptr @load_ptr(ptr %t3716, i64 1)
  %t3718 = ptrtoint ptr %t3717 to i64
  store i64 %t3718, ptr %local.name.3313
  %t3719 = load i64, ptr %local.expr
  %t3720 = inttoptr i64 %t3719 to ptr
  %t3721 = call ptr @load_ptr(ptr %t3720, i64 2)
  %t3722 = ptrtoint ptr %t3721 to i64
  store i64 %t3722, ptr %local.fields.3314
  %t3723 = call ptr @intrinsic_sb_new()
  %t3724 = ptrtoint ptr %t3723 to i64
  store i64 %t3724, ptr %local.sb.3315
  %t3725 = load i64, ptr %local.sb.3315
  %t3726 = load i64, ptr %local.name.3313
  %t3727 = inttoptr i64 %t3725 to ptr
  %t3728 = inttoptr i64 %t3726 to ptr
  call void @intrinsic_sb_append(ptr %t3727, ptr %t3728)
  %t3729 = load i64, ptr %local.sb.3315
  %t3730 = call ptr @intrinsic_string_new(ptr @.str.main.192)
  %t3731 = ptrtoint ptr %t3730 to i64
  %t3732 = inttoptr i64 %t3729 to ptr
  %t3733 = inttoptr i64 %t3731 to ptr
  call void @intrinsic_sb_append(ptr %t3732, ptr %t3733)
  store i64 0, ptr %local.i.3316
  br label %loop231
loop231:
  %t3734 = load i64, ptr %local.i.3316
  %t3735 = load i64, ptr %local.fields.3314
  %t3736 = inttoptr i64 %t3735 to ptr
  %t3737 = call i64 @intrinsic_vec_len(ptr %t3736)
  %t3738 = icmp slt i64 %t3734, %t3737
  %t3739 = zext i1 %t3738 to i64
  %t3740 = icmp ne i64 %t3739, 0
  br i1 %t3740, label %body231, label %endloop231
body231:
  %t3741 = load i64, ptr %local.i.3316
  %t3742 = icmp sgt i64 %t3741, 0
  %t3743 = zext i1 %t3742 to i64
  %t3744 = icmp ne i64 %t3743, 0
  br i1 %t3744, label %then232, label %else232
then232:
  %t3745 = load i64, ptr %local.sb.3315
  %t3746 = call ptr @intrinsic_string_new(ptr @.str.main.193)
  %t3747 = ptrtoint ptr %t3746 to i64
  %t3748 = inttoptr i64 %t3745 to ptr
  %t3749 = inttoptr i64 %t3747 to ptr
  call void @intrinsic_sb_append(ptr %t3748, ptr %t3749)
  br label %then232_end
then232_end:
  br label %endif232
else232:
  br label %else232_end
else232_end:
  br label %endif232
endif232:
  %t3750 = phi i64 [ 0, %then232_end ], [ 0, %else232_end ]
  %t3751 = load i64, ptr %local.fields.3314
  %t3752 = load i64, ptr %local.i.3316
  %t3753 = inttoptr i64 %t3751 to ptr
  %t3754 = call ptr @intrinsic_vec_get(ptr %t3753, i64 %t3752)
  %t3755 = ptrtoint ptr %t3754 to i64
  store i64 %t3755, ptr %local.field.3317
  %t3756 = load i64, ptr %local.field.3317
  %t3757 = inttoptr i64 %t3756 to ptr
  %t3758 = call ptr @load_ptr(ptr %t3757, i64 0)
  %t3759 = ptrtoint ptr %t3758 to i64
  store i64 %t3759, ptr %local.fname.3318
  %t3760 = load i64, ptr %local.field.3317
  %t3761 = inttoptr i64 %t3760 to ptr
  %t3762 = call ptr @load_ptr(ptr %t3761, i64 1)
  %t3763 = ptrtoint ptr %t3762 to i64
  store i64 %t3763, ptr %local.fval.3319
  %t3764 = load i64, ptr %local.sb.3315
  %t3765 = load i64, ptr %local.fname.3318
  %t3766 = inttoptr i64 %t3764 to ptr
  %t3767 = inttoptr i64 %t3765 to ptr
  call void @intrinsic_sb_append(ptr %t3766, ptr %t3767)
  %t3768 = load i64, ptr %local.sb.3315
  %t3769 = call ptr @intrinsic_string_new(ptr @.str.main.194)
  %t3770 = ptrtoint ptr %t3769 to i64
  %t3771 = inttoptr i64 %t3768 to ptr
  %t3772 = inttoptr i64 %t3770 to ptr
  call void @intrinsic_sb_append(ptr %t3771, ptr %t3772)
  %t3773 = load i64, ptr %local.sb.3315
  %t3774 = load i64, ptr %local.fval.3319
  %t3775 = load i64, ptr %local.indent
  %t3776 = call i64 @"format_expr"(i64 %t3774, i64 %t3775)
  %t3777 = inttoptr i64 %t3773 to ptr
  %t3778 = inttoptr i64 %t3776 to ptr
  call void @intrinsic_sb_append(ptr %t3777, ptr %t3778)
  %t3779 = load i64, ptr %local.i.3316
  %t3780 = add i64 %t3779, 1
  store i64 %t3780, ptr %local.i.3316
  br label %loop231
endloop231:
  %t3781 = load i64, ptr %local.sb.3315
  %t3782 = call ptr @intrinsic_string_new(ptr @.str.main.195)
  %t3783 = ptrtoint ptr %t3782 to i64
  %t3784 = inttoptr i64 %t3781 to ptr
  %t3785 = inttoptr i64 %t3783 to ptr
  call void @intrinsic_sb_append(ptr %t3784, ptr %t3785)
  %t3786 = load i64, ptr %local.sb.3315
  %t3787 = inttoptr i64 %t3786 to ptr
  %t3788 = call ptr @intrinsic_sb_to_string(ptr %t3787)
  %t3789 = ptrtoint ptr %t3788 to i64
  ret i64 %t3789
  br label %then230_end
then230_end:
  br label %endif230
else230:
  br label %else230_end
else230_end:
  br label %endif230
endif230:
  %t3790 = phi i64 [ 0, %then230_end ], [ 0, %else230_end ]
  %t3791 = load i64, ptr %local.tag.3290
  %t3792 = icmp eq i64 %t3791, 12
  %t3793 = zext i1 %t3792 to i64
  %t3794 = icmp ne i64 %t3793, 0
  br i1 %t3794, label %then233, label %else233
then233:
  %t3795 = load i64, ptr %local.expr
  %t3796 = inttoptr i64 %t3795 to ptr
  %t3797 = call ptr @load_ptr(ptr %t3796, i64 1)
  %t3798 = ptrtoint ptr %t3797 to i64
  store i64 %t3798, ptr %local.obj.3320
  %t3799 = load i64, ptr %local.expr
  %t3800 = inttoptr i64 %t3799 to ptr
  %t3801 = call ptr @load_ptr(ptr %t3800, i64 2)
  %t3802 = ptrtoint ptr %t3801 to i64
  store i64 %t3802, ptr %local.field.3321
  %t3803 = load i64, ptr %local.obj.3320
  %t3804 = load i64, ptr %local.indent
  %t3805 = call i64 @"format_expr"(i64 %t3803, i64 %t3804)
  %t3806 = call ptr @intrinsic_string_new(ptr @.str.main.196)
  %t3807 = ptrtoint ptr %t3806 to i64
  %t3808 = load i64, ptr %local.field.3321
  %t3809 = inttoptr i64 %t3807 to ptr
  %t3810 = inttoptr i64 %t3808 to ptr
  %t3811 = call ptr @intrinsic_string_concat(ptr %t3809, ptr %t3810)
  %t3812 = ptrtoint ptr %t3811 to i64
  %t3813 = inttoptr i64 %t3805 to ptr
  %t3814 = inttoptr i64 %t3812 to ptr
  %t3815 = call ptr @intrinsic_string_concat(ptr %t3813, ptr %t3814)
  %t3816 = ptrtoint ptr %t3815 to i64
  ret i64 %t3816
  br label %then233_end
then233_end:
  br label %endif233
else233:
  br label %else233_end
else233_end:
  br label %endif233
endif233:
  %t3817 = phi i64 [ 0, %then233_end ], [ 0, %else233_end ]
  %t3818 = load i64, ptr %local.tag.3290
  %t3819 = icmp eq i64 %t3818, 13
  %t3820 = zext i1 %t3819 to i64
  %t3821 = icmp ne i64 %t3820, 0
  br i1 %t3821, label %then234, label %else234
then234:
  %t3822 = load i64, ptr %local.expr
  %t3823 = inttoptr i64 %t3822 to ptr
  %t3824 = call ptr @load_ptr(ptr %t3823, i64 1)
  %t3825 = ptrtoint ptr %t3824 to i64
  store i64 %t3825, ptr %local.loop_var.3322
  %t3826 = load i64, ptr %local.expr
  %t3827 = inttoptr i64 %t3826 to ptr
  %t3828 = call ptr @load_ptr(ptr %t3827, i64 2)
  %t3829 = ptrtoint ptr %t3828 to i64
  store i64 %t3829, ptr %local.start.3323
  %t3830 = load i64, ptr %local.expr
  %t3831 = inttoptr i64 %t3830 to ptr
  %t3832 = call ptr @load_ptr(ptr %t3831, i64 3)
  %t3833 = ptrtoint ptr %t3832 to i64
  store i64 %t3833, ptr %local.end.3324
  %t3834 = load i64, ptr %local.expr
  %t3835 = inttoptr i64 %t3834 to ptr
  %t3836 = call ptr @load_ptr(ptr %t3835, i64 4)
  %t3837 = ptrtoint ptr %t3836 to i64
  store i64 %t3837, ptr %local.body.3325
  %t3838 = call ptr @intrinsic_string_new(ptr @.str.main.197)
  %t3839 = ptrtoint ptr %t3838 to i64
  %t3840 = load i64, ptr %local.loop_var.3322
  %t3841 = call ptr @intrinsic_string_new(ptr @.str.main.198)
  %t3842 = ptrtoint ptr %t3841 to i64
  %t3843 = load i64, ptr %local.start.3323
  %t3844 = load i64, ptr %local.indent
  %t3845 = call i64 @"format_expr"(i64 %t3843, i64 %t3844)
  %t3846 = call ptr @intrinsic_string_new(ptr @.str.main.199)
  %t3847 = ptrtoint ptr %t3846 to i64
  %t3848 = load i64, ptr %local.end.3324
  %t3849 = load i64, ptr %local.indent
  %t3850 = call i64 @"format_expr"(i64 %t3848, i64 %t3849)
  %t3851 = call ptr @intrinsic_string_new(ptr @.str.main.200)
  %t3852 = ptrtoint ptr %t3851 to i64
  %t3853 = load i64, ptr %local.body.3325
  %t3854 = load i64, ptr %local.indent
  %t3855 = call i64 @"format_block"(i64 %t3853, i64 %t3854)
  %t3856 = inttoptr i64 %t3852 to ptr
  %t3857 = inttoptr i64 %t3855 to ptr
  %t3858 = call ptr @intrinsic_string_concat(ptr %t3856, ptr %t3857)
  %t3859 = ptrtoint ptr %t3858 to i64
  %t3860 = inttoptr i64 %t3850 to ptr
  %t3861 = inttoptr i64 %t3859 to ptr
  %t3862 = call ptr @intrinsic_string_concat(ptr %t3860, ptr %t3861)
  %t3863 = ptrtoint ptr %t3862 to i64
  %t3864 = inttoptr i64 %t3847 to ptr
  %t3865 = inttoptr i64 %t3863 to ptr
  %t3866 = call ptr @intrinsic_string_concat(ptr %t3864, ptr %t3865)
  %t3867 = ptrtoint ptr %t3866 to i64
  %t3868 = inttoptr i64 %t3845 to ptr
  %t3869 = inttoptr i64 %t3867 to ptr
  %t3870 = call ptr @intrinsic_string_concat(ptr %t3868, ptr %t3869)
  %t3871 = ptrtoint ptr %t3870 to i64
  %t3872 = inttoptr i64 %t3842 to ptr
  %t3873 = inttoptr i64 %t3871 to ptr
  %t3874 = call ptr @intrinsic_string_concat(ptr %t3872, ptr %t3873)
  %t3875 = ptrtoint ptr %t3874 to i64
  %t3876 = inttoptr i64 %t3840 to ptr
  %t3877 = inttoptr i64 %t3875 to ptr
  %t3878 = call ptr @intrinsic_string_concat(ptr %t3876, ptr %t3877)
  %t3879 = ptrtoint ptr %t3878 to i64
  %t3880 = inttoptr i64 %t3839 to ptr
  %t3881 = inttoptr i64 %t3879 to ptr
  %t3882 = call ptr @intrinsic_string_concat(ptr %t3880, ptr %t3881)
  %t3883 = ptrtoint ptr %t3882 to i64
  ret i64 %t3883
  br label %then234_end
then234_end:
  br label %endif234
else234:
  br label %else234_end
else234_end:
  br label %endif234
endif234:
  %t3884 = phi i64 [ 0, %then234_end ], [ 0, %else234_end ]
  %t3885 = load i64, ptr %local.tag.3290
  %t3886 = icmp eq i64 %t3885, 14
  %t3887 = zext i1 %t3886 to i64
  %t3888 = icmp ne i64 %t3887, 0
  br i1 %t3888, label %then235, label %else235
then235:
  %t3889 = load i64, ptr %local.expr
  %t3890 = inttoptr i64 %t3889 to ptr
  %t3891 = call ptr @load_ptr(ptr %t3890, i64 1)
  %t3892 = ptrtoint ptr %t3891 to i64
  store i64 %t3892, ptr %local.scrutinee.3326
  %t3893 = load i64, ptr %local.expr
  %t3894 = inttoptr i64 %t3893 to ptr
  %t3895 = call ptr @load_ptr(ptr %t3894, i64 2)
  %t3896 = ptrtoint ptr %t3895 to i64
  store i64 %t3896, ptr %local.arms.3327
  %t3897 = call ptr @intrinsic_sb_new()
  %t3898 = ptrtoint ptr %t3897 to i64
  store i64 %t3898, ptr %local.sb.3328
  %t3899 = load i64, ptr %local.sb.3328
  %t3900 = call ptr @intrinsic_string_new(ptr @.str.main.201)
  %t3901 = ptrtoint ptr %t3900 to i64
  %t3902 = inttoptr i64 %t3899 to ptr
  %t3903 = inttoptr i64 %t3901 to ptr
  call void @intrinsic_sb_append(ptr %t3902, ptr %t3903)
  %t3904 = load i64, ptr %local.sb.3328
  %t3905 = load i64, ptr %local.scrutinee.3326
  %t3906 = load i64, ptr %local.indent
  %t3907 = call i64 @"format_expr"(i64 %t3905, i64 %t3906)
  %t3908 = inttoptr i64 %t3904 to ptr
  %t3909 = inttoptr i64 %t3907 to ptr
  call void @intrinsic_sb_append(ptr %t3908, ptr %t3909)
  %t3910 = load i64, ptr %local.sb.3328
  %t3911 = call ptr @intrinsic_string_new(ptr @.str.main.202)
  %t3912 = ptrtoint ptr %t3911 to i64
  %t3913 = inttoptr i64 %t3910 to ptr
  %t3914 = inttoptr i64 %t3912 to ptr
  call void @intrinsic_sb_append(ptr %t3913, ptr %t3914)
  store i64 0, ptr %local.i.3329
  br label %loop236
loop236:
  %t3915 = load i64, ptr %local.i.3329
  %t3916 = load i64, ptr %local.arms.3327
  %t3917 = inttoptr i64 %t3916 to ptr
  %t3918 = call i64 @intrinsic_vec_len(ptr %t3917)
  %t3919 = icmp slt i64 %t3915, %t3918
  %t3920 = zext i1 %t3919 to i64
  %t3921 = icmp ne i64 %t3920, 0
  br i1 %t3921, label %body236, label %endloop236
body236:
  %t3922 = load i64, ptr %local.arms.3327
  %t3923 = load i64, ptr %local.i.3329
  %t3924 = inttoptr i64 %t3922 to ptr
  %t3925 = call ptr @intrinsic_vec_get(ptr %t3924, i64 %t3923)
  %t3926 = ptrtoint ptr %t3925 to i64
  store i64 %t3926, ptr %local.arm.3330
  %t3927 = load i64, ptr %local.arm.3330
  %t3928 = inttoptr i64 %t3927 to ptr
  %t3929 = call ptr @load_ptr(ptr %t3928, i64 0)
  %t3930 = ptrtoint ptr %t3929 to i64
  store i64 %t3930, ptr %local.pattern.3331
  %t3931 = load i64, ptr %local.arm.3330
  %t3932 = inttoptr i64 %t3931 to ptr
  %t3933 = call ptr @load_ptr(ptr %t3932, i64 1)
  %t3934 = ptrtoint ptr %t3933 to i64
  store i64 %t3934, ptr %local.result.3332
  %t3935 = load i64, ptr %local.sb.3328
  %t3936 = load i64, ptr %local.indent
  %t3937 = add i64 %t3936, 1
  %t3938 = call i64 @"make_indent"(i64 %t3937)
  %t3939 = inttoptr i64 %t3935 to ptr
  %t3940 = inttoptr i64 %t3938 to ptr
  call void @intrinsic_sb_append(ptr %t3939, ptr %t3940)
  %t3941 = load i64, ptr %local.pattern.3331
  %t3942 = icmp eq i64 %t3941, 0
  %t3943 = zext i1 %t3942 to i64
  %t3944 = icmp ne i64 %t3943, 0
  br i1 %t3944, label %then237, label %else237
then237:
  %t3945 = load i64, ptr %local.sb.3328
  %t3946 = call ptr @intrinsic_string_new(ptr @.str.main.203)
  %t3947 = ptrtoint ptr %t3946 to i64
  %t3948 = inttoptr i64 %t3945 to ptr
  %t3949 = inttoptr i64 %t3947 to ptr
  call void @intrinsic_sb_append(ptr %t3948, ptr %t3949)
  br label %then237_end
then237_end:
  br label %endif237
else237:
  %t3950 = load i64, ptr %local.sb.3328
  %t3951 = load i64, ptr %local.pattern.3331
  %t3952 = load i64, ptr %local.indent
  %t3953 = add i64 %t3952, 1
  %t3954 = call i64 @"format_expr"(i64 %t3951, i64 %t3953)
  %t3955 = inttoptr i64 %t3950 to ptr
  %t3956 = inttoptr i64 %t3954 to ptr
  call void @intrinsic_sb_append(ptr %t3955, ptr %t3956)
  br label %else237_end
else237_end:
  br label %endif237
endif237:
  %t3957 = phi i64 [ 0, %then237_end ], [ 0, %else237_end ]
  %t3958 = load i64, ptr %local.sb.3328
  %t3959 = call ptr @intrinsic_string_new(ptr @.str.main.204)
  %t3960 = ptrtoint ptr %t3959 to i64
  %t3961 = inttoptr i64 %t3958 to ptr
  %t3962 = inttoptr i64 %t3960 to ptr
  call void @intrinsic_sb_append(ptr %t3961, ptr %t3962)
  %t3963 = load i64, ptr %local.sb.3328
  %t3964 = load i64, ptr %local.result.3332
  %t3965 = load i64, ptr %local.indent
  %t3966 = add i64 %t3965, 1
  %t3967 = call i64 @"format_expr"(i64 %t3964, i64 %t3966)
  %t3968 = inttoptr i64 %t3963 to ptr
  %t3969 = inttoptr i64 %t3967 to ptr
  call void @intrinsic_sb_append(ptr %t3968, ptr %t3969)
  %t3970 = load i64, ptr %local.sb.3328
  %t3971 = call ptr @intrinsic_string_new(ptr @.str.main.205)
  %t3972 = ptrtoint ptr %t3971 to i64
  %t3973 = inttoptr i64 %t3970 to ptr
  %t3974 = inttoptr i64 %t3972 to ptr
  call void @intrinsic_sb_append(ptr %t3973, ptr %t3974)
  %t3975 = load i64, ptr %local.i.3329
  %t3976 = add i64 %t3975, 1
  store i64 %t3976, ptr %local.i.3329
  br label %loop236
endloop236:
  %t3977 = load i64, ptr %local.sb.3328
  %t3978 = load i64, ptr %local.indent
  %t3979 = call i64 @"make_indent"(i64 %t3978)
  %t3980 = inttoptr i64 %t3977 to ptr
  %t3981 = inttoptr i64 %t3979 to ptr
  call void @intrinsic_sb_append(ptr %t3980, ptr %t3981)
  %t3982 = load i64, ptr %local.sb.3328
  %t3983 = call ptr @intrinsic_string_new(ptr @.str.main.206)
  %t3984 = ptrtoint ptr %t3983 to i64
  %t3985 = inttoptr i64 %t3982 to ptr
  %t3986 = inttoptr i64 %t3984 to ptr
  call void @intrinsic_sb_append(ptr %t3985, ptr %t3986)
  %t3987 = load i64, ptr %local.sb.3328
  %t3988 = inttoptr i64 %t3987 to ptr
  %t3989 = call ptr @intrinsic_sb_to_string(ptr %t3988)
  %t3990 = ptrtoint ptr %t3989 to i64
  ret i64 %t3990
  br label %then235_end
then235_end:
  br label %endif235
else235:
  br label %else235_end
else235_end:
  br label %endif235
endif235:
  %t3991 = phi i64 [ 0, %then235_end ], [ 0, %else235_end ]
  %t3992 = load i64, ptr %local.tag.3290
  %t3993 = icmp eq i64 %t3992, 15
  %t3994 = zext i1 %t3993 to i64
  %t3995 = icmp ne i64 %t3994, 0
  br i1 %t3995, label %then238, label %else238
then238:
  %t3996 = load i64, ptr %local.expr
  %t3997 = inttoptr i64 %t3996 to ptr
  %t3998 = call ptr @load_ptr(ptr %t3997, i64 1)
  %t3999 = ptrtoint ptr %t3998 to i64
  store i64 %t3999, ptr %local.obj.3333
  %t4000 = load i64, ptr %local.expr
  %t4001 = inttoptr i64 %t4000 to ptr
  %t4002 = call ptr @load_ptr(ptr %t4001, i64 2)
  %t4003 = ptrtoint ptr %t4002 to i64
  store i64 %t4003, ptr %local.method.3334
  %t4004 = load i64, ptr %local.expr
  %t4005 = inttoptr i64 %t4004 to ptr
  %t4006 = call ptr @load_ptr(ptr %t4005, i64 3)
  %t4007 = ptrtoint ptr %t4006 to i64
  store i64 %t4007, ptr %local.args.3335
  %t4008 = call ptr @intrinsic_sb_new()
  %t4009 = ptrtoint ptr %t4008 to i64
  store i64 %t4009, ptr %local.sb.3336
  %t4010 = load i64, ptr %local.sb.3336
  %t4011 = load i64, ptr %local.obj.3333
  %t4012 = load i64, ptr %local.indent
  %t4013 = call i64 @"format_expr"(i64 %t4011, i64 %t4012)
  %t4014 = inttoptr i64 %t4010 to ptr
  %t4015 = inttoptr i64 %t4013 to ptr
  call void @intrinsic_sb_append(ptr %t4014, ptr %t4015)
  %t4016 = load i64, ptr %local.sb.3336
  %t4017 = call ptr @intrinsic_string_new(ptr @.str.main.207)
  %t4018 = ptrtoint ptr %t4017 to i64
  %t4019 = inttoptr i64 %t4016 to ptr
  %t4020 = inttoptr i64 %t4018 to ptr
  call void @intrinsic_sb_append(ptr %t4019, ptr %t4020)
  %t4021 = load i64, ptr %local.sb.3336
  %t4022 = load i64, ptr %local.method.3334
  %t4023 = inttoptr i64 %t4021 to ptr
  %t4024 = inttoptr i64 %t4022 to ptr
  call void @intrinsic_sb_append(ptr %t4023, ptr %t4024)
  %t4025 = load i64, ptr %local.sb.3336
  %t4026 = call ptr @intrinsic_string_new(ptr @.str.main.208)
  %t4027 = ptrtoint ptr %t4026 to i64
  %t4028 = inttoptr i64 %t4025 to ptr
  %t4029 = inttoptr i64 %t4027 to ptr
  call void @intrinsic_sb_append(ptr %t4028, ptr %t4029)
  store i64 0, ptr %local.i.3337
  br label %loop239
loop239:
  %t4030 = load i64, ptr %local.i.3337
  %t4031 = load i64, ptr %local.args.3335
  %t4032 = inttoptr i64 %t4031 to ptr
  %t4033 = call i64 @intrinsic_vec_len(ptr %t4032)
  %t4034 = icmp slt i64 %t4030, %t4033
  %t4035 = zext i1 %t4034 to i64
  %t4036 = icmp ne i64 %t4035, 0
  br i1 %t4036, label %body239, label %endloop239
body239:
  %t4037 = load i64, ptr %local.i.3337
  %t4038 = icmp sgt i64 %t4037, 0
  %t4039 = zext i1 %t4038 to i64
  %t4040 = icmp ne i64 %t4039, 0
  br i1 %t4040, label %then240, label %else240
then240:
  %t4041 = load i64, ptr %local.sb.3336
  %t4042 = call ptr @intrinsic_string_new(ptr @.str.main.209)
  %t4043 = ptrtoint ptr %t4042 to i64
  %t4044 = inttoptr i64 %t4041 to ptr
  %t4045 = inttoptr i64 %t4043 to ptr
  call void @intrinsic_sb_append(ptr %t4044, ptr %t4045)
  br label %then240_end
then240_end:
  br label %endif240
else240:
  br label %else240_end
else240_end:
  br label %endif240
endif240:
  %t4046 = phi i64 [ 0, %then240_end ], [ 0, %else240_end ]
  %t4047 = load i64, ptr %local.sb.3336
  %t4048 = load i64, ptr %local.args.3335
  %t4049 = load i64, ptr %local.i.3337
  %t4050 = inttoptr i64 %t4048 to ptr
  %t4051 = call ptr @intrinsic_vec_get(ptr %t4050, i64 %t4049)
  %t4052 = ptrtoint ptr %t4051 to i64
  %t4053 = load i64, ptr %local.indent
  %t4054 = call i64 @"format_expr"(i64 %t4052, i64 %t4053)
  %t4055 = inttoptr i64 %t4047 to ptr
  %t4056 = inttoptr i64 %t4054 to ptr
  call void @intrinsic_sb_append(ptr %t4055, ptr %t4056)
  %t4057 = load i64, ptr %local.i.3337
  %t4058 = add i64 %t4057, 1
  store i64 %t4058, ptr %local.i.3337
  br label %loop239
endloop239:
  %t4059 = load i64, ptr %local.sb.3336
  %t4060 = call ptr @intrinsic_string_new(ptr @.str.main.210)
  %t4061 = ptrtoint ptr %t4060 to i64
  %t4062 = inttoptr i64 %t4059 to ptr
  %t4063 = inttoptr i64 %t4061 to ptr
  call void @intrinsic_sb_append(ptr %t4062, ptr %t4063)
  %t4064 = load i64, ptr %local.sb.3336
  %t4065 = inttoptr i64 %t4064 to ptr
  %t4066 = call ptr @intrinsic_sb_to_string(ptr %t4065)
  %t4067 = ptrtoint ptr %t4066 to i64
  ret i64 %t4067
  br label %then238_end
then238_end:
  br label %endif238
else238:
  br label %else238_end
else238_end:
  br label %endif238
endif238:
  %t4068 = phi i64 [ 0, %then238_end ], [ 0, %else238_end ]
  %t4069 = load i64, ptr %local.tag.3290
  %t4070 = icmp eq i64 %t4069, 16
  %t4071 = zext i1 %t4070 to i64
  %t4072 = icmp ne i64 %t4071, 0
  br i1 %t4072, label %then241, label %else241
then241:
  %t4073 = load i64, ptr %local.expr
  %t4074 = inttoptr i64 %t4073 to ptr
  %t4075 = call ptr @load_ptr(ptr %t4074, i64 1)
  %t4076 = ptrtoint ptr %t4075 to i64
  store i64 %t4076, ptr %local.inner.3338
  %t4077 = load i64, ptr %local.inner.3338
  %t4078 = load i64, ptr %local.indent
  %t4079 = call i64 @"format_expr"(i64 %t4077, i64 %t4078)
  %t4080 = call ptr @intrinsic_string_new(ptr @.str.main.211)
  %t4081 = ptrtoint ptr %t4080 to i64
  %t4082 = inttoptr i64 %t4079 to ptr
  %t4083 = inttoptr i64 %t4081 to ptr
  %t4084 = call ptr @intrinsic_string_concat(ptr %t4082, ptr %t4083)
  %t4085 = ptrtoint ptr %t4084 to i64
  ret i64 %t4085
  br label %then241_end
then241_end:
  br label %endif241
else241:
  br label %else241_end
else241_end:
  br label %endif241
endif241:
  %t4086 = phi i64 [ 0, %then241_end ], [ 0, %else241_end ]
  %t4087 = call ptr @intrinsic_string_new(ptr @.str.main.212)
  %t4088 = ptrtoint ptr %t4087 to i64
  ret i64 %t4088
}

define i64 @"format_binop"(i64 %op) {
entry:
  %local.op = alloca i64
  store i64 %op, ptr %local.op
  %t4089 = load i64, ptr %local.op
  %t4090 = icmp eq i64 %t4089, 0
  %t4091 = zext i1 %t4090 to i64
  %t4092 = icmp ne i64 %t4091, 0
  br i1 %t4092, label %then242, label %else242
then242:
  %t4093 = call ptr @intrinsic_string_new(ptr @.str.main.213)
  %t4094 = ptrtoint ptr %t4093 to i64
  ret i64 %t4094
  br label %then242_end
then242_end:
  br label %endif242
else242:
  br label %else242_end
else242_end:
  br label %endif242
endif242:
  %t4095 = phi i64 [ 0, %then242_end ], [ 0, %else242_end ]
  %t4096 = load i64, ptr %local.op
  %t4097 = icmp eq i64 %t4096, 1
  %t4098 = zext i1 %t4097 to i64
  %t4099 = icmp ne i64 %t4098, 0
  br i1 %t4099, label %then243, label %else243
then243:
  %t4100 = call ptr @intrinsic_string_new(ptr @.str.main.214)
  %t4101 = ptrtoint ptr %t4100 to i64
  ret i64 %t4101
  br label %then243_end
then243_end:
  br label %endif243
else243:
  br label %else243_end
else243_end:
  br label %endif243
endif243:
  %t4102 = phi i64 [ 0, %then243_end ], [ 0, %else243_end ]
  %t4103 = load i64, ptr %local.op
  %t4104 = icmp eq i64 %t4103, 2
  %t4105 = zext i1 %t4104 to i64
  %t4106 = icmp ne i64 %t4105, 0
  br i1 %t4106, label %then244, label %else244
then244:
  %t4107 = call ptr @intrinsic_string_new(ptr @.str.main.215)
  %t4108 = ptrtoint ptr %t4107 to i64
  ret i64 %t4108
  br label %then244_end
then244_end:
  br label %endif244
else244:
  br label %else244_end
else244_end:
  br label %endif244
endif244:
  %t4109 = phi i64 [ 0, %then244_end ], [ 0, %else244_end ]
  %t4110 = load i64, ptr %local.op
  %t4111 = icmp eq i64 %t4110, 3
  %t4112 = zext i1 %t4111 to i64
  %t4113 = icmp ne i64 %t4112, 0
  br i1 %t4113, label %then245, label %else245
then245:
  %t4114 = call ptr @intrinsic_string_new(ptr @.str.main.216)
  %t4115 = ptrtoint ptr %t4114 to i64
  ret i64 %t4115
  br label %then245_end
then245_end:
  br label %endif245
else245:
  br label %else245_end
else245_end:
  br label %endif245
endif245:
  %t4116 = phi i64 [ 0, %then245_end ], [ 0, %else245_end ]
  %t4117 = load i64, ptr %local.op
  %t4118 = icmp eq i64 %t4117, 4
  %t4119 = zext i1 %t4118 to i64
  %t4120 = icmp ne i64 %t4119, 0
  br i1 %t4120, label %then246, label %else246
then246:
  %t4121 = call ptr @intrinsic_string_new(ptr @.str.main.217)
  %t4122 = ptrtoint ptr %t4121 to i64
  ret i64 %t4122
  br label %then246_end
then246_end:
  br label %endif246
else246:
  br label %else246_end
else246_end:
  br label %endif246
endif246:
  %t4123 = phi i64 [ 0, %then246_end ], [ 0, %else246_end ]
  %t4124 = load i64, ptr %local.op
  %t4125 = icmp eq i64 %t4124, 5
  %t4126 = zext i1 %t4125 to i64
  %t4127 = icmp ne i64 %t4126, 0
  br i1 %t4127, label %then247, label %else247
then247:
  %t4128 = call ptr @intrinsic_string_new(ptr @.str.main.218)
  %t4129 = ptrtoint ptr %t4128 to i64
  ret i64 %t4129
  br label %then247_end
then247_end:
  br label %endif247
else247:
  br label %else247_end
else247_end:
  br label %endif247
endif247:
  %t4130 = phi i64 [ 0, %then247_end ], [ 0, %else247_end ]
  %t4131 = load i64, ptr %local.op
  %t4132 = icmp eq i64 %t4131, 6
  %t4133 = zext i1 %t4132 to i64
  %t4134 = icmp ne i64 %t4133, 0
  br i1 %t4134, label %then248, label %else248
then248:
  %t4135 = call ptr @intrinsic_string_new(ptr @.str.main.219)
  %t4136 = ptrtoint ptr %t4135 to i64
  ret i64 %t4136
  br label %then248_end
then248_end:
  br label %endif248
else248:
  br label %else248_end
else248_end:
  br label %endif248
endif248:
  %t4137 = phi i64 [ 0, %then248_end ], [ 0, %else248_end ]
  %t4138 = load i64, ptr %local.op
  %t4139 = icmp eq i64 %t4138, 7
  %t4140 = zext i1 %t4139 to i64
  %t4141 = icmp ne i64 %t4140, 0
  br i1 %t4141, label %then249, label %else249
then249:
  %t4142 = call ptr @intrinsic_string_new(ptr @.str.main.220)
  %t4143 = ptrtoint ptr %t4142 to i64
  ret i64 %t4143
  br label %then249_end
then249_end:
  br label %endif249
else249:
  br label %else249_end
else249_end:
  br label %endif249
endif249:
  %t4144 = phi i64 [ 0, %then249_end ], [ 0, %else249_end ]
  %t4145 = load i64, ptr %local.op
  %t4146 = icmp eq i64 %t4145, 8
  %t4147 = zext i1 %t4146 to i64
  %t4148 = icmp ne i64 %t4147, 0
  br i1 %t4148, label %then250, label %else250
then250:
  %t4149 = call ptr @intrinsic_string_new(ptr @.str.main.221)
  %t4150 = ptrtoint ptr %t4149 to i64
  ret i64 %t4150
  br label %then250_end
then250_end:
  br label %endif250
else250:
  br label %else250_end
else250_end:
  br label %endif250
endif250:
  %t4151 = phi i64 [ 0, %then250_end ], [ 0, %else250_end ]
  %t4152 = load i64, ptr %local.op
  %t4153 = icmp eq i64 %t4152, 9
  %t4154 = zext i1 %t4153 to i64
  %t4155 = icmp ne i64 %t4154, 0
  br i1 %t4155, label %then251, label %else251
then251:
  %t4156 = call ptr @intrinsic_string_new(ptr @.str.main.222)
  %t4157 = ptrtoint ptr %t4156 to i64
  ret i64 %t4157
  br label %then251_end
then251_end:
  br label %endif251
else251:
  br label %else251_end
else251_end:
  br label %endif251
endif251:
  %t4158 = phi i64 [ 0, %then251_end ], [ 0, %else251_end ]
  %t4159 = load i64, ptr %local.op
  %t4160 = icmp eq i64 %t4159, 10
  %t4161 = zext i1 %t4160 to i64
  %t4162 = icmp ne i64 %t4161, 0
  br i1 %t4162, label %then252, label %else252
then252:
  %t4163 = call ptr @intrinsic_string_new(ptr @.str.main.223)
  %t4164 = ptrtoint ptr %t4163 to i64
  ret i64 %t4164
  br label %then252_end
then252_end:
  br label %endif252
else252:
  br label %else252_end
else252_end:
  br label %endif252
endif252:
  %t4165 = phi i64 [ 0, %then252_end ], [ 0, %else252_end ]
  %t4166 = load i64, ptr %local.op
  %t4167 = icmp eq i64 %t4166, 11
  %t4168 = zext i1 %t4167 to i64
  %t4169 = icmp ne i64 %t4168, 0
  br i1 %t4169, label %then253, label %else253
then253:
  %t4170 = call ptr @intrinsic_string_new(ptr @.str.main.224)
  %t4171 = ptrtoint ptr %t4170 to i64
  ret i64 %t4171
  br label %then253_end
then253_end:
  br label %endif253
else253:
  br label %else253_end
else253_end:
  br label %endif253
endif253:
  %t4172 = phi i64 [ 0, %then253_end ], [ 0, %else253_end ]
  %t4173 = load i64, ptr %local.op
  %t4174 = icmp eq i64 %t4173, 12
  %t4175 = zext i1 %t4174 to i64
  %t4176 = icmp ne i64 %t4175, 0
  br i1 %t4176, label %then254, label %else254
then254:
  %t4177 = call ptr @intrinsic_string_new(ptr @.str.main.225)
  %t4178 = ptrtoint ptr %t4177 to i64
  ret i64 %t4178
  br label %then254_end
then254_end:
  br label %endif254
else254:
  br label %else254_end
else254_end:
  br label %endif254
endif254:
  %t4179 = phi i64 [ 0, %then254_end ], [ 0, %else254_end ]
  %t4180 = load i64, ptr %local.op
  %t4181 = icmp eq i64 %t4180, 13
  %t4182 = zext i1 %t4181 to i64
  %t4183 = icmp ne i64 %t4182, 0
  br i1 %t4183, label %then255, label %else255
then255:
  %t4184 = call ptr @intrinsic_string_new(ptr @.str.main.226)
  %t4185 = ptrtoint ptr %t4184 to i64
  ret i64 %t4185
  br label %then255_end
then255_end:
  br label %endif255
else255:
  br label %else255_end
else255_end:
  br label %endif255
endif255:
  %t4186 = phi i64 [ 0, %then255_end ], [ 0, %else255_end ]
  %t4187 = load i64, ptr %local.op
  %t4188 = icmp eq i64 %t4187, 14
  %t4189 = zext i1 %t4188 to i64
  %t4190 = icmp ne i64 %t4189, 0
  br i1 %t4190, label %then256, label %else256
then256:
  %t4191 = call ptr @intrinsic_string_new(ptr @.str.main.227)
  %t4192 = ptrtoint ptr %t4191 to i64
  ret i64 %t4192
  br label %then256_end
then256_end:
  br label %endif256
else256:
  br label %else256_end
else256_end:
  br label %endif256
endif256:
  %t4193 = phi i64 [ 0, %then256_end ], [ 0, %else256_end ]
  %t4194 = load i64, ptr %local.op
  %t4195 = icmp eq i64 %t4194, 15
  %t4196 = zext i1 %t4195 to i64
  %t4197 = icmp ne i64 %t4196, 0
  br i1 %t4197, label %then257, label %else257
then257:
  %t4198 = call ptr @intrinsic_string_new(ptr @.str.main.228)
  %t4199 = ptrtoint ptr %t4198 to i64
  ret i64 %t4199
  br label %then257_end
then257_end:
  br label %endif257
else257:
  br label %else257_end
else257_end:
  br label %endif257
endif257:
  %t4200 = phi i64 [ 0, %then257_end ], [ 0, %else257_end ]
  %t4201 = load i64, ptr %local.op
  %t4202 = icmp eq i64 %t4201, 16
  %t4203 = zext i1 %t4202 to i64
  %t4204 = icmp ne i64 %t4203, 0
  br i1 %t4204, label %then258, label %else258
then258:
  %t4205 = call ptr @intrinsic_string_new(ptr @.str.main.229)
  %t4206 = ptrtoint ptr %t4205 to i64
  ret i64 %t4206
  br label %then258_end
then258_end:
  br label %endif258
else258:
  br label %else258_end
else258_end:
  br label %endif258
endif258:
  %t4207 = phi i64 [ 0, %then258_end ], [ 0, %else258_end ]
  %t4208 = load i64, ptr %local.op
  %t4209 = icmp eq i64 %t4208, 17
  %t4210 = zext i1 %t4209 to i64
  %t4211 = icmp ne i64 %t4210, 0
  br i1 %t4211, label %then259, label %else259
then259:
  %t4212 = call ptr @intrinsic_string_new(ptr @.str.main.230)
  %t4213 = ptrtoint ptr %t4212 to i64
  ret i64 %t4213
  br label %then259_end
then259_end:
  br label %endif259
else259:
  br label %else259_end
else259_end:
  br label %endif259
endif259:
  %t4214 = phi i64 [ 0, %then259_end ], [ 0, %else259_end ]
  %t4215 = call ptr @intrinsic_string_new(ptr @.str.main.231)
  %t4216 = ptrtoint ptr %t4215 to i64
  ret i64 %t4216
}

define i64 @"format_enum"(i64 %enum_def) {
entry:
  %local.name.4217 = alloca i64
  %local.variants.4218 = alloca i64
  %local.sb.4219 = alloca i64
  %local.i.4220 = alloca i64
  %local.enum_def = alloca i64
  store i64 %enum_def, ptr %local.enum_def
  %t4221 = load i64, ptr %local.enum_def
  %t4222 = inttoptr i64 %t4221 to ptr
  %t4223 = call ptr @load_ptr(ptr %t4222, i64 1)
  %t4224 = ptrtoint ptr %t4223 to i64
  store i64 %t4224, ptr %local.name.4217
  %t4225 = load i64, ptr %local.enum_def
  %t4226 = inttoptr i64 %t4225 to ptr
  %t4227 = call ptr @load_ptr(ptr %t4226, i64 2)
  %t4228 = ptrtoint ptr %t4227 to i64
  store i64 %t4228, ptr %local.variants.4218
  %t4229 = call ptr @intrinsic_sb_new()
  %t4230 = ptrtoint ptr %t4229 to i64
  store i64 %t4230, ptr %local.sb.4219
  %t4231 = load i64, ptr %local.sb.4219
  %t4232 = call ptr @intrinsic_string_new(ptr @.str.main.232)
  %t4233 = ptrtoint ptr %t4232 to i64
  %t4234 = inttoptr i64 %t4231 to ptr
  %t4235 = inttoptr i64 %t4233 to ptr
  call void @intrinsic_sb_append(ptr %t4234, ptr %t4235)
  %t4236 = load i64, ptr %local.sb.4219
  %t4237 = load i64, ptr %local.name.4217
  %t4238 = inttoptr i64 %t4236 to ptr
  %t4239 = inttoptr i64 %t4237 to ptr
  call void @intrinsic_sb_append(ptr %t4238, ptr %t4239)
  %t4240 = load i64, ptr %local.sb.4219
  %t4241 = call ptr @intrinsic_string_new(ptr @.str.main.233)
  %t4242 = ptrtoint ptr %t4241 to i64
  %t4243 = inttoptr i64 %t4240 to ptr
  %t4244 = inttoptr i64 %t4242 to ptr
  call void @intrinsic_sb_append(ptr %t4243, ptr %t4244)
  store i64 0, ptr %local.i.4220
  br label %loop260
loop260:
  %t4245 = load i64, ptr %local.i.4220
  %t4246 = load i64, ptr %local.variants.4218
  %t4247 = inttoptr i64 %t4246 to ptr
  %t4248 = call i64 @intrinsic_vec_len(ptr %t4247)
  %t4249 = icmp slt i64 %t4245, %t4248
  %t4250 = zext i1 %t4249 to i64
  %t4251 = icmp ne i64 %t4250, 0
  br i1 %t4251, label %body260, label %endloop260
body260:
  %t4252 = load i64, ptr %local.sb.4219
  %t4253 = call ptr @intrinsic_string_new(ptr @.str.main.234)
  %t4254 = ptrtoint ptr %t4253 to i64
  %t4255 = inttoptr i64 %t4252 to ptr
  %t4256 = inttoptr i64 %t4254 to ptr
  call void @intrinsic_sb_append(ptr %t4255, ptr %t4256)
  %t4257 = load i64, ptr %local.sb.4219
  %t4258 = load i64, ptr %local.variants.4218
  %t4259 = load i64, ptr %local.i.4220
  %t4260 = inttoptr i64 %t4258 to ptr
  %t4261 = call ptr @intrinsic_vec_get(ptr %t4260, i64 %t4259)
  %t4262 = ptrtoint ptr %t4261 to i64
  %t4263 = inttoptr i64 %t4257 to ptr
  %t4264 = inttoptr i64 %t4262 to ptr
  call void @intrinsic_sb_append(ptr %t4263, ptr %t4264)
  %t4265 = load i64, ptr %local.i.4220
  %t4266 = load i64, ptr %local.variants.4218
  %t4267 = inttoptr i64 %t4266 to ptr
  %t4268 = call i64 @intrinsic_vec_len(ptr %t4267)
  %t4269 = sub i64 %t4268, 1
  %t4270 = icmp slt i64 %t4265, %t4269
  %t4271 = zext i1 %t4270 to i64
  %t4272 = icmp ne i64 %t4271, 0
  br i1 %t4272, label %then261, label %else261
then261:
  %t4273 = load i64, ptr %local.sb.4219
  %t4274 = call ptr @intrinsic_string_new(ptr @.str.main.235)
  %t4275 = ptrtoint ptr %t4274 to i64
  %t4276 = inttoptr i64 %t4273 to ptr
  %t4277 = inttoptr i64 %t4275 to ptr
  call void @intrinsic_sb_append(ptr %t4276, ptr %t4277)
  br label %then261_end
then261_end:
  br label %endif261
else261:
  br label %else261_end
else261_end:
  br label %endif261
endif261:
  %t4278 = phi i64 [ 0, %then261_end ], [ 0, %else261_end ]
  %t4279 = load i64, ptr %local.sb.4219
  %t4280 = call ptr @intrinsic_string_new(ptr @.str.main.236)
  %t4281 = ptrtoint ptr %t4280 to i64
  %t4282 = inttoptr i64 %t4279 to ptr
  %t4283 = inttoptr i64 %t4281 to ptr
  call void @intrinsic_sb_append(ptr %t4282, ptr %t4283)
  %t4284 = load i64, ptr %local.i.4220
  %t4285 = add i64 %t4284, 1
  store i64 %t4285, ptr %local.i.4220
  br label %loop260
endloop260:
  %t4286 = load i64, ptr %local.sb.4219
  %t4287 = call ptr @intrinsic_string_new(ptr @.str.main.237)
  %t4288 = ptrtoint ptr %t4287 to i64
  %t4289 = inttoptr i64 %t4286 to ptr
  %t4290 = inttoptr i64 %t4288 to ptr
  call void @intrinsic_sb_append(ptr %t4289, ptr %t4290)
  %t4291 = load i64, ptr %local.sb.4219
  %t4292 = inttoptr i64 %t4291 to ptr
  %t4293 = call ptr @intrinsic_sb_to_string(ptr %t4292)
  %t4294 = ptrtoint ptr %t4293 to i64
  ret i64 %t4294
}

define i64 @"format_struct"(i64 %struct_def) {
entry:
  %local.name.4295 = alloca i64
  %local.type_params.4296 = alloca i64
  %local.fields.4297 = alloca i64
  %local.sb.4298 = alloca i64
  %local.i.4299 = alloca i64
  %local.i.4300 = alloca i64
  %local.field.4301 = alloca i64
  %local.fname.4302 = alloca i64
  %local.fty.4303 = alloca i64
  %local.struct_def = alloca i64
  store i64 %struct_def, ptr %local.struct_def
  %t4304 = load i64, ptr %local.struct_def
  %t4305 = inttoptr i64 %t4304 to ptr
  %t4306 = call ptr @load_ptr(ptr %t4305, i64 1)
  %t4307 = ptrtoint ptr %t4306 to i64
  store i64 %t4307, ptr %local.name.4295
  %t4308 = load i64, ptr %local.struct_def
  %t4309 = inttoptr i64 %t4308 to ptr
  %t4310 = call ptr @load_ptr(ptr %t4309, i64 2)
  %t4311 = ptrtoint ptr %t4310 to i64
  store i64 %t4311, ptr %local.type_params.4296
  %t4312 = load i64, ptr %local.struct_def
  %t4313 = inttoptr i64 %t4312 to ptr
  %t4314 = call ptr @load_ptr(ptr %t4313, i64 3)
  %t4315 = ptrtoint ptr %t4314 to i64
  store i64 %t4315, ptr %local.fields.4297
  %t4316 = call ptr @intrinsic_sb_new()
  %t4317 = ptrtoint ptr %t4316 to i64
  store i64 %t4317, ptr %local.sb.4298
  %t4318 = load i64, ptr %local.sb.4298
  %t4319 = call ptr @intrinsic_string_new(ptr @.str.main.238)
  %t4320 = ptrtoint ptr %t4319 to i64
  %t4321 = inttoptr i64 %t4318 to ptr
  %t4322 = inttoptr i64 %t4320 to ptr
  call void @intrinsic_sb_append(ptr %t4321, ptr %t4322)
  %t4323 = load i64, ptr %local.sb.4298
  %t4324 = load i64, ptr %local.name.4295
  %t4325 = inttoptr i64 %t4323 to ptr
  %t4326 = inttoptr i64 %t4324 to ptr
  call void @intrinsic_sb_append(ptr %t4325, ptr %t4326)
  %t4327 = load i64, ptr %local.type_params.4296
  %t4328 = icmp ne i64 %t4327, 0
  %t4329 = zext i1 %t4328 to i64
  %t4330 = icmp ne i64 %t4329, 0
  br i1 %t4330, label %then262, label %else262
then262:
  %t4331 = load i64, ptr %local.type_params.4296
  %t4332 = inttoptr i64 %t4331 to ptr
  %t4333 = call i64 @intrinsic_vec_len(ptr %t4332)
  %t4334 = icmp sgt i64 %t4333, 0
  %t4335 = zext i1 %t4334 to i64
  %t4336 = icmp ne i64 %t4335, 0
  br i1 %t4336, label %then263, label %else263
then263:
  %t4337 = load i64, ptr %local.sb.4298
  %t4338 = call ptr @intrinsic_string_new(ptr @.str.main.239)
  %t4339 = ptrtoint ptr %t4338 to i64
  %t4340 = inttoptr i64 %t4337 to ptr
  %t4341 = inttoptr i64 %t4339 to ptr
  call void @intrinsic_sb_append(ptr %t4340, ptr %t4341)
  store i64 0, ptr %local.i.4299
  br label %loop264
loop264:
  %t4342 = load i64, ptr %local.i.4299
  %t4343 = load i64, ptr %local.type_params.4296
  %t4344 = inttoptr i64 %t4343 to ptr
  %t4345 = call i64 @intrinsic_vec_len(ptr %t4344)
  %t4346 = icmp slt i64 %t4342, %t4345
  %t4347 = zext i1 %t4346 to i64
  %t4348 = icmp ne i64 %t4347, 0
  br i1 %t4348, label %body264, label %endloop264
body264:
  %t4349 = load i64, ptr %local.i.4299
  %t4350 = icmp sgt i64 %t4349, 0
  %t4351 = zext i1 %t4350 to i64
  %t4352 = icmp ne i64 %t4351, 0
  br i1 %t4352, label %then265, label %else265
then265:
  %t4353 = load i64, ptr %local.sb.4298
  %t4354 = call ptr @intrinsic_string_new(ptr @.str.main.240)
  %t4355 = ptrtoint ptr %t4354 to i64
  %t4356 = inttoptr i64 %t4353 to ptr
  %t4357 = inttoptr i64 %t4355 to ptr
  call void @intrinsic_sb_append(ptr %t4356, ptr %t4357)
  br label %then265_end
then265_end:
  br label %endif265
else265:
  br label %else265_end
else265_end:
  br label %endif265
endif265:
  %t4358 = phi i64 [ 0, %then265_end ], [ 0, %else265_end ]
  %t4359 = load i64, ptr %local.sb.4298
  %t4360 = load i64, ptr %local.type_params.4296
  %t4361 = load i64, ptr %local.i.4299
  %t4362 = inttoptr i64 %t4360 to ptr
  %t4363 = call ptr @intrinsic_vec_get(ptr %t4362, i64 %t4361)
  %t4364 = ptrtoint ptr %t4363 to i64
  %t4365 = inttoptr i64 %t4359 to ptr
  %t4366 = inttoptr i64 %t4364 to ptr
  call void @intrinsic_sb_append(ptr %t4365, ptr %t4366)
  %t4367 = load i64, ptr %local.i.4299
  %t4368 = add i64 %t4367, 1
  store i64 %t4368, ptr %local.i.4299
  br label %loop264
endloop264:
  %t4369 = load i64, ptr %local.sb.4298
  %t4370 = call ptr @intrinsic_string_new(ptr @.str.main.241)
  %t4371 = ptrtoint ptr %t4370 to i64
  %t4372 = inttoptr i64 %t4369 to ptr
  %t4373 = inttoptr i64 %t4371 to ptr
  call void @intrinsic_sb_append(ptr %t4372, ptr %t4373)
  br label %then263_end
then263_end:
  br label %endif263
else263:
  br label %else263_end
else263_end:
  br label %endif263
endif263:
  %t4374 = phi i64 [ 0, %then263_end ], [ 0, %else263_end ]
  br label %then262_end
then262_end:
  br label %endif262
else262:
  br label %else262_end
else262_end:
  br label %endif262
endif262:
  %t4375 = phi i64 [ %t4374, %then262_end ], [ 0, %else262_end ]
  %t4376 = load i64, ptr %local.sb.4298
  %t4377 = call ptr @intrinsic_string_new(ptr @.str.main.242)
  %t4378 = ptrtoint ptr %t4377 to i64
  %t4379 = inttoptr i64 %t4376 to ptr
  %t4380 = inttoptr i64 %t4378 to ptr
  call void @intrinsic_sb_append(ptr %t4379, ptr %t4380)
  store i64 0, ptr %local.i.4300
  br label %loop266
loop266:
  %t4381 = load i64, ptr %local.i.4300
  %t4382 = load i64, ptr %local.fields.4297
  %t4383 = inttoptr i64 %t4382 to ptr
  %t4384 = call i64 @intrinsic_vec_len(ptr %t4383)
  %t4385 = icmp slt i64 %t4381, %t4384
  %t4386 = zext i1 %t4385 to i64
  %t4387 = icmp ne i64 %t4386, 0
  br i1 %t4387, label %body266, label %endloop266
body266:
  %t4388 = load i64, ptr %local.fields.4297
  %t4389 = load i64, ptr %local.i.4300
  %t4390 = inttoptr i64 %t4388 to ptr
  %t4391 = call ptr @intrinsic_vec_get(ptr %t4390, i64 %t4389)
  %t4392 = ptrtoint ptr %t4391 to i64
  store i64 %t4392, ptr %local.field.4301
  %t4393 = load i64, ptr %local.field.4301
  %t4394 = inttoptr i64 %t4393 to ptr
  %t4395 = call ptr @load_ptr(ptr %t4394, i64 0)
  %t4396 = ptrtoint ptr %t4395 to i64
  store i64 %t4396, ptr %local.fname.4302
  %t4397 = load i64, ptr %local.field.4301
  %t4398 = inttoptr i64 %t4397 to ptr
  %t4399 = call ptr @load_ptr(ptr %t4398, i64 1)
  %t4400 = ptrtoint ptr %t4399 to i64
  store i64 %t4400, ptr %local.fty.4303
  %t4401 = load i64, ptr %local.sb.4298
  %t4402 = call ptr @intrinsic_string_new(ptr @.str.main.243)
  %t4403 = ptrtoint ptr %t4402 to i64
  %t4404 = inttoptr i64 %t4401 to ptr
  %t4405 = inttoptr i64 %t4403 to ptr
  call void @intrinsic_sb_append(ptr %t4404, ptr %t4405)
  %t4406 = load i64, ptr %local.sb.4298
  %t4407 = load i64, ptr %local.fname.4302
  %t4408 = inttoptr i64 %t4406 to ptr
  %t4409 = inttoptr i64 %t4407 to ptr
  call void @intrinsic_sb_append(ptr %t4408, ptr %t4409)
  %t4410 = load i64, ptr %local.sb.4298
  %t4411 = call ptr @intrinsic_string_new(ptr @.str.main.244)
  %t4412 = ptrtoint ptr %t4411 to i64
  %t4413 = inttoptr i64 %t4410 to ptr
  %t4414 = inttoptr i64 %t4412 to ptr
  call void @intrinsic_sb_append(ptr %t4413, ptr %t4414)
  %t4415 = load i64, ptr %local.sb.4298
  %t4416 = load i64, ptr %local.fty.4303
  %t4417 = inttoptr i64 %t4415 to ptr
  %t4418 = inttoptr i64 %t4416 to ptr
  call void @intrinsic_sb_append(ptr %t4417, ptr %t4418)
  %t4419 = load i64, ptr %local.i.4300
  %t4420 = load i64, ptr %local.fields.4297
  %t4421 = inttoptr i64 %t4420 to ptr
  %t4422 = call i64 @intrinsic_vec_len(ptr %t4421)
  %t4423 = sub i64 %t4422, 1
  %t4424 = icmp slt i64 %t4419, %t4423
  %t4425 = zext i1 %t4424 to i64
  %t4426 = icmp ne i64 %t4425, 0
  br i1 %t4426, label %then267, label %else267
then267:
  %t4427 = load i64, ptr %local.sb.4298
  %t4428 = call ptr @intrinsic_string_new(ptr @.str.main.245)
  %t4429 = ptrtoint ptr %t4428 to i64
  %t4430 = inttoptr i64 %t4427 to ptr
  %t4431 = inttoptr i64 %t4429 to ptr
  call void @intrinsic_sb_append(ptr %t4430, ptr %t4431)
  br label %then267_end
then267_end:
  br label %endif267
else267:
  br label %else267_end
else267_end:
  br label %endif267
endif267:
  %t4432 = phi i64 [ 0, %then267_end ], [ 0, %else267_end ]
  %t4433 = load i64, ptr %local.sb.4298
  %t4434 = call ptr @intrinsic_string_new(ptr @.str.main.246)
  %t4435 = ptrtoint ptr %t4434 to i64
  %t4436 = inttoptr i64 %t4433 to ptr
  %t4437 = inttoptr i64 %t4435 to ptr
  call void @intrinsic_sb_append(ptr %t4436, ptr %t4437)
  %t4438 = load i64, ptr %local.i.4300
  %t4439 = add i64 %t4438, 1
  store i64 %t4439, ptr %local.i.4300
  br label %loop266
endloop266:
  %t4440 = load i64, ptr %local.sb.4298
  %t4441 = call ptr @intrinsic_string_new(ptr @.str.main.247)
  %t4442 = ptrtoint ptr %t4441 to i64
  %t4443 = inttoptr i64 %t4440 to ptr
  %t4444 = inttoptr i64 %t4442 to ptr
  call void @intrinsic_sb_append(ptr %t4443, ptr %t4444)
  %t4445 = load i64, ptr %local.sb.4298
  %t4446 = inttoptr i64 %t4445 to ptr
  %t4447 = call ptr @intrinsic_sb_to_string(ptr %t4446)
  %t4448 = ptrtoint ptr %t4447 to i64
  ret i64 %t4448
}

define i64 @"format_impl"(i64 %impl_def) {
entry:
  %local.type_name.4449 = alloca i64
  %local.methods.4450 = alloca i64
  %local.sb.4451 = alloca i64
  %local.i.4452 = alloca i64
  %local.method.4453 = alloca i64
  %local.formatted.4454 = alloca i64
  %local.impl_def = alloca i64
  store i64 %impl_def, ptr %local.impl_def
  %t4455 = load i64, ptr %local.impl_def
  %t4456 = inttoptr i64 %t4455 to ptr
  %t4457 = call ptr @load_ptr(ptr %t4456, i64 1)
  %t4458 = ptrtoint ptr %t4457 to i64
  store i64 %t4458, ptr %local.type_name.4449
  %t4459 = load i64, ptr %local.impl_def
  %t4460 = inttoptr i64 %t4459 to ptr
  %t4461 = call ptr @load_ptr(ptr %t4460, i64 2)
  %t4462 = ptrtoint ptr %t4461 to i64
  store i64 %t4462, ptr %local.methods.4450
  %t4463 = call ptr @intrinsic_sb_new()
  %t4464 = ptrtoint ptr %t4463 to i64
  store i64 %t4464, ptr %local.sb.4451
  %t4465 = load i64, ptr %local.sb.4451
  %t4466 = call ptr @intrinsic_string_new(ptr @.str.main.248)
  %t4467 = ptrtoint ptr %t4466 to i64
  %t4468 = inttoptr i64 %t4465 to ptr
  %t4469 = inttoptr i64 %t4467 to ptr
  call void @intrinsic_sb_append(ptr %t4468, ptr %t4469)
  %t4470 = load i64, ptr %local.sb.4451
  %t4471 = load i64, ptr %local.type_name.4449
  %t4472 = inttoptr i64 %t4470 to ptr
  %t4473 = inttoptr i64 %t4471 to ptr
  call void @intrinsic_sb_append(ptr %t4472, ptr %t4473)
  %t4474 = load i64, ptr %local.sb.4451
  %t4475 = call ptr @intrinsic_string_new(ptr @.str.main.249)
  %t4476 = ptrtoint ptr %t4475 to i64
  %t4477 = inttoptr i64 %t4474 to ptr
  %t4478 = inttoptr i64 %t4476 to ptr
  call void @intrinsic_sb_append(ptr %t4477, ptr %t4478)
  store i64 0, ptr %local.i.4452
  br label %loop268
loop268:
  %t4479 = load i64, ptr %local.i.4452
  %t4480 = load i64, ptr %local.methods.4450
  %t4481 = inttoptr i64 %t4480 to ptr
  %t4482 = call i64 @intrinsic_vec_len(ptr %t4481)
  %t4483 = icmp slt i64 %t4479, %t4482
  %t4484 = zext i1 %t4483 to i64
  %t4485 = icmp ne i64 %t4484, 0
  br i1 %t4485, label %body268, label %endloop268
body268:
  %t4486 = load i64, ptr %local.methods.4450
  %t4487 = load i64, ptr %local.i.4452
  %t4488 = inttoptr i64 %t4486 to ptr
  %t4489 = call ptr @intrinsic_vec_get(ptr %t4488, i64 %t4487)
  %t4490 = ptrtoint ptr %t4489 to i64
  store i64 %t4490, ptr %local.method.4453
  %t4491 = load i64, ptr %local.sb.4451
  %t4492 = call ptr @intrinsic_string_new(ptr @.str.main.250)
  %t4493 = ptrtoint ptr %t4492 to i64
  %t4494 = inttoptr i64 %t4491 to ptr
  %t4495 = inttoptr i64 %t4493 to ptr
  call void @intrinsic_sb_append(ptr %t4494, ptr %t4495)
  %t4496 = load i64, ptr %local.method.4453
  %t4497 = call i64 @"format_fn"(i64 %t4496)
  store i64 %t4497, ptr %local.formatted.4454
  %t4498 = load i64, ptr %local.formatted.4454
  %t4499 = call ptr @intrinsic_string_new(ptr @.str.main.251)
  %t4500 = ptrtoint ptr %t4499 to i64
  %t4501 = call ptr @intrinsic_string_new(ptr @.str.main.252)
  %t4502 = ptrtoint ptr %t4501 to i64
  %t4503 = inttoptr i64 %t4498 to ptr
  %t4504 = inttoptr i64 %t4500 to ptr
  %t4505 = inttoptr i64 %t4502 to ptr
  %t4506 = call ptr @intrinsic_string_replace(ptr %t4503, ptr %t4504, ptr %t4505)
  %t4507 = ptrtoint ptr %t4506 to i64
  store i64 %t4507, ptr %local.formatted.4454
  %t4508 = load i64, ptr %local.sb.4451
  %t4509 = load i64, ptr %local.formatted.4454
  %t4510 = inttoptr i64 %t4508 to ptr
  %t4511 = inttoptr i64 %t4509 to ptr
  call void @intrinsic_sb_append(ptr %t4510, ptr %t4511)
  %t4512 = load i64, ptr %local.sb.4451
  %t4513 = call ptr @intrinsic_string_new(ptr @.str.main.253)
  %t4514 = ptrtoint ptr %t4513 to i64
  %t4515 = inttoptr i64 %t4512 to ptr
  %t4516 = inttoptr i64 %t4514 to ptr
  call void @intrinsic_sb_append(ptr %t4515, ptr %t4516)
  %t4517 = load i64, ptr %local.i.4452
  %t4518 = add i64 %t4517, 1
  store i64 %t4518, ptr %local.i.4452
  br label %loop268
endloop268:
  %t4519 = load i64, ptr %local.sb.4451
  %t4520 = call ptr @intrinsic_string_new(ptr @.str.main.254)
  %t4521 = ptrtoint ptr %t4520 to i64
  %t4522 = inttoptr i64 %t4519 to ptr
  %t4523 = inttoptr i64 %t4521 to ptr
  call void @intrinsic_sb_append(ptr %t4522, ptr %t4523)
  %t4524 = load i64, ptr %local.sb.4451
  %t4525 = inttoptr i64 %t4524 to ptr
  %t4526 = call ptr @intrinsic_sb_to_string(ptr %t4525)
  %t4527 = ptrtoint ptr %t4526 to i64
  ret i64 %t4527
}

define i64 @"format_trait"(i64 %trait_def) {
entry:
  %local.name.4528 = alloca i64
  %local.method_sigs.4529 = alloca i64
  %local.sb.4530 = alloca i64
  %local.i.4531 = alloca i64
  %local.sig.4532 = alloca i64
  %local.mname.4533 = alloca i64
  %local.params.4534 = alloca i64
  %local.ret_ty.4535 = alloca i64
  %local.j.4536 = alloca i64
  %local.param.4537 = alloca i64
  %local.pname.4538 = alloca i64
  %local.pty.4539 = alloca i64
  %local.trait_def = alloca i64
  store i64 %trait_def, ptr %local.trait_def
  %t4540 = load i64, ptr %local.trait_def
  %t4541 = inttoptr i64 %t4540 to ptr
  %t4542 = call ptr @load_ptr(ptr %t4541, i64 1)
  %t4543 = ptrtoint ptr %t4542 to i64
  store i64 %t4543, ptr %local.name.4528
  %t4544 = load i64, ptr %local.trait_def
  %t4545 = inttoptr i64 %t4544 to ptr
  %t4546 = call ptr @load_ptr(ptr %t4545, i64 2)
  %t4547 = ptrtoint ptr %t4546 to i64
  store i64 %t4547, ptr %local.method_sigs.4529
  %t4548 = call ptr @intrinsic_sb_new()
  %t4549 = ptrtoint ptr %t4548 to i64
  store i64 %t4549, ptr %local.sb.4530
  %t4550 = load i64, ptr %local.sb.4530
  %t4551 = call ptr @intrinsic_string_new(ptr @.str.main.255)
  %t4552 = ptrtoint ptr %t4551 to i64
  %t4553 = inttoptr i64 %t4550 to ptr
  %t4554 = inttoptr i64 %t4552 to ptr
  call void @intrinsic_sb_append(ptr %t4553, ptr %t4554)
  %t4555 = load i64, ptr %local.sb.4530
  %t4556 = load i64, ptr %local.name.4528
  %t4557 = inttoptr i64 %t4555 to ptr
  %t4558 = inttoptr i64 %t4556 to ptr
  call void @intrinsic_sb_append(ptr %t4557, ptr %t4558)
  %t4559 = load i64, ptr %local.sb.4530
  %t4560 = call ptr @intrinsic_string_new(ptr @.str.main.256)
  %t4561 = ptrtoint ptr %t4560 to i64
  %t4562 = inttoptr i64 %t4559 to ptr
  %t4563 = inttoptr i64 %t4561 to ptr
  call void @intrinsic_sb_append(ptr %t4562, ptr %t4563)
  store i64 0, ptr %local.i.4531
  br label %loop269
loop269:
  %t4564 = load i64, ptr %local.i.4531
  %t4565 = load i64, ptr %local.method_sigs.4529
  %t4566 = inttoptr i64 %t4565 to ptr
  %t4567 = call i64 @intrinsic_vec_len(ptr %t4566)
  %t4568 = icmp slt i64 %t4564, %t4567
  %t4569 = zext i1 %t4568 to i64
  %t4570 = icmp ne i64 %t4569, 0
  br i1 %t4570, label %body269, label %endloop269
body269:
  %t4571 = load i64, ptr %local.method_sigs.4529
  %t4572 = load i64, ptr %local.i.4531
  %t4573 = inttoptr i64 %t4571 to ptr
  %t4574 = call ptr @intrinsic_vec_get(ptr %t4573, i64 %t4572)
  %t4575 = ptrtoint ptr %t4574 to i64
  store i64 %t4575, ptr %local.sig.4532
  %t4576 = load i64, ptr %local.sig.4532
  %t4577 = inttoptr i64 %t4576 to ptr
  %t4578 = call ptr @load_ptr(ptr %t4577, i64 0)
  %t4579 = ptrtoint ptr %t4578 to i64
  store i64 %t4579, ptr %local.mname.4533
  %t4580 = load i64, ptr %local.sig.4532
  %t4581 = inttoptr i64 %t4580 to ptr
  %t4582 = call ptr @load_ptr(ptr %t4581, i64 1)
  %t4583 = ptrtoint ptr %t4582 to i64
  store i64 %t4583, ptr %local.params.4534
  %t4584 = load i64, ptr %local.sig.4532
  %t4585 = inttoptr i64 %t4584 to ptr
  %t4586 = call ptr @load_ptr(ptr %t4585, i64 2)
  %t4587 = ptrtoint ptr %t4586 to i64
  store i64 %t4587, ptr %local.ret_ty.4535
  %t4588 = load i64, ptr %local.sb.4530
  %t4589 = call ptr @intrinsic_string_new(ptr @.str.main.257)
  %t4590 = ptrtoint ptr %t4589 to i64
  %t4591 = inttoptr i64 %t4588 to ptr
  %t4592 = inttoptr i64 %t4590 to ptr
  call void @intrinsic_sb_append(ptr %t4591, ptr %t4592)
  %t4593 = load i64, ptr %local.sb.4530
  %t4594 = load i64, ptr %local.mname.4533
  %t4595 = inttoptr i64 %t4593 to ptr
  %t4596 = inttoptr i64 %t4594 to ptr
  call void @intrinsic_sb_append(ptr %t4595, ptr %t4596)
  %t4597 = load i64, ptr %local.sb.4530
  %t4598 = call ptr @intrinsic_string_new(ptr @.str.main.258)
  %t4599 = ptrtoint ptr %t4598 to i64
  %t4600 = inttoptr i64 %t4597 to ptr
  %t4601 = inttoptr i64 %t4599 to ptr
  call void @intrinsic_sb_append(ptr %t4600, ptr %t4601)
  %t4602 = load i64, ptr %local.params.4534
  %t4603 = icmp ne i64 %t4602, 0
  %t4604 = zext i1 %t4603 to i64
  %t4605 = icmp ne i64 %t4604, 0
  br i1 %t4605, label %then270, label %else270
then270:
  store i64 0, ptr %local.j.4536
  br label %loop271
loop271:
  %t4606 = load i64, ptr %local.j.4536
  %t4607 = load i64, ptr %local.params.4534
  %t4608 = inttoptr i64 %t4607 to ptr
  %t4609 = call i64 @intrinsic_vec_len(ptr %t4608)
  %t4610 = icmp slt i64 %t4606, %t4609
  %t4611 = zext i1 %t4610 to i64
  %t4612 = icmp ne i64 %t4611, 0
  br i1 %t4612, label %body271, label %endloop271
body271:
  %t4613 = load i64, ptr %local.j.4536
  %t4614 = icmp sgt i64 %t4613, 0
  %t4615 = zext i1 %t4614 to i64
  %t4616 = icmp ne i64 %t4615, 0
  br i1 %t4616, label %then272, label %else272
then272:
  %t4617 = load i64, ptr %local.sb.4530
  %t4618 = call ptr @intrinsic_string_new(ptr @.str.main.259)
  %t4619 = ptrtoint ptr %t4618 to i64
  %t4620 = inttoptr i64 %t4617 to ptr
  %t4621 = inttoptr i64 %t4619 to ptr
  call void @intrinsic_sb_append(ptr %t4620, ptr %t4621)
  br label %then272_end
then272_end:
  br label %endif272
else272:
  br label %else272_end
else272_end:
  br label %endif272
endif272:
  %t4622 = phi i64 [ 0, %then272_end ], [ 0, %else272_end ]
  %t4623 = load i64, ptr %local.params.4534
  %t4624 = load i64, ptr %local.j.4536
  %t4625 = inttoptr i64 %t4623 to ptr
  %t4626 = call ptr @intrinsic_vec_get(ptr %t4625, i64 %t4624)
  %t4627 = ptrtoint ptr %t4626 to i64
  store i64 %t4627, ptr %local.param.4537
  %t4628 = load i64, ptr %local.param.4537
  %t4629 = inttoptr i64 %t4628 to ptr
  %t4630 = call ptr @load_ptr(ptr %t4629, i64 0)
  %t4631 = ptrtoint ptr %t4630 to i64
  store i64 %t4631, ptr %local.pname.4538
  %t4632 = load i64, ptr %local.param.4537
  %t4633 = inttoptr i64 %t4632 to ptr
  %t4634 = call ptr @load_ptr(ptr %t4633, i64 1)
  %t4635 = ptrtoint ptr %t4634 to i64
  store i64 %t4635, ptr %local.pty.4539
  %t4636 = load i64, ptr %local.sb.4530
  %t4637 = load i64, ptr %local.pname.4538
  %t4638 = inttoptr i64 %t4636 to ptr
  %t4639 = inttoptr i64 %t4637 to ptr
  call void @intrinsic_sb_append(ptr %t4638, ptr %t4639)
  %t4640 = load i64, ptr %local.sb.4530
  %t4641 = call ptr @intrinsic_string_new(ptr @.str.main.260)
  %t4642 = ptrtoint ptr %t4641 to i64
  %t4643 = inttoptr i64 %t4640 to ptr
  %t4644 = inttoptr i64 %t4642 to ptr
  call void @intrinsic_sb_append(ptr %t4643, ptr %t4644)
  %t4645 = load i64, ptr %local.sb.4530
  %t4646 = load i64, ptr %local.pty.4539
  %t4647 = inttoptr i64 %t4645 to ptr
  %t4648 = inttoptr i64 %t4646 to ptr
  call void @intrinsic_sb_append(ptr %t4647, ptr %t4648)
  %t4649 = load i64, ptr %local.j.4536
  %t4650 = add i64 %t4649, 1
  store i64 %t4650, ptr %local.j.4536
  br label %loop271
endloop271:
  br label %then270_end
then270_end:
  br label %endif270
else270:
  br label %else270_end
else270_end:
  br label %endif270
endif270:
  %t4651 = phi i64 [ 0, %then270_end ], [ 0, %else270_end ]
  %t4652 = load i64, ptr %local.sb.4530
  %t4653 = call ptr @intrinsic_string_new(ptr @.str.main.261)
  %t4654 = ptrtoint ptr %t4653 to i64
  %t4655 = inttoptr i64 %t4652 to ptr
  %t4656 = inttoptr i64 %t4654 to ptr
  call void @intrinsic_sb_append(ptr %t4655, ptr %t4656)
  %t4657 = load i64, ptr %local.ret_ty.4535
  %t4658 = icmp ne i64 %t4657, 0
  %t4659 = zext i1 %t4658 to i64
  %t4660 = icmp ne i64 %t4659, 0
  br i1 %t4660, label %then273, label %else273
then273:
  %t4661 = load i64, ptr %local.sb.4530
  %t4662 = call ptr @intrinsic_string_new(ptr @.str.main.262)
  %t4663 = ptrtoint ptr %t4662 to i64
  %t4664 = inttoptr i64 %t4661 to ptr
  %t4665 = inttoptr i64 %t4663 to ptr
  call void @intrinsic_sb_append(ptr %t4664, ptr %t4665)
  %t4666 = load i64, ptr %local.sb.4530
  %t4667 = load i64, ptr %local.ret_ty.4535
  %t4668 = inttoptr i64 %t4666 to ptr
  %t4669 = inttoptr i64 %t4667 to ptr
  call void @intrinsic_sb_append(ptr %t4668, ptr %t4669)
  br label %then273_end
then273_end:
  br label %endif273
else273:
  br label %else273_end
else273_end:
  br label %endif273
endif273:
  %t4670 = phi i64 [ 0, %then273_end ], [ 0, %else273_end ]
  %t4671 = load i64, ptr %local.sb.4530
  %t4672 = call ptr @intrinsic_string_new(ptr @.str.main.263)
  %t4673 = ptrtoint ptr %t4672 to i64
  %t4674 = inttoptr i64 %t4671 to ptr
  %t4675 = inttoptr i64 %t4673 to ptr
  call void @intrinsic_sb_append(ptr %t4674, ptr %t4675)
  %t4676 = load i64, ptr %local.i.4531
  %t4677 = add i64 %t4676, 1
  store i64 %t4677, ptr %local.i.4531
  br label %loop269
endloop269:
  %t4678 = load i64, ptr %local.sb.4530
  %t4679 = call ptr @intrinsic_string_new(ptr @.str.main.264)
  %t4680 = ptrtoint ptr %t4679 to i64
  %t4681 = inttoptr i64 %t4678 to ptr
  %t4682 = inttoptr i64 %t4680 to ptr
  call void @intrinsic_sb_append(ptr %t4681, ptr %t4682)
  %t4683 = load i64, ptr %local.sb.4530
  %t4684 = inttoptr i64 %t4683 to ptr
  %t4685 = call ptr @intrinsic_sb_to_string(ptr %t4684)
  %t4686 = ptrtoint ptr %t4685 to i64
  ret i64 %t4686
}

define i64 @"format_impl_trait"(i64 %impl_def) {
entry:
  %local.trait_name.4687 = alloca i64
  %local.type_name.4688 = alloca i64
  %local.methods.4689 = alloca i64
  %local.sb.4690 = alloca i64
  %local.i.4691 = alloca i64
  %local.method.4692 = alloca i64
  %local.formatted.4693 = alloca i64
  %local.impl_def = alloca i64
  store i64 %impl_def, ptr %local.impl_def
  %t4694 = load i64, ptr %local.impl_def
  %t4695 = inttoptr i64 %t4694 to ptr
  %t4696 = call ptr @load_ptr(ptr %t4695, i64 1)
  %t4697 = ptrtoint ptr %t4696 to i64
  store i64 %t4697, ptr %local.trait_name.4687
  %t4698 = load i64, ptr %local.impl_def
  %t4699 = inttoptr i64 %t4698 to ptr
  %t4700 = call ptr @load_ptr(ptr %t4699, i64 2)
  %t4701 = ptrtoint ptr %t4700 to i64
  store i64 %t4701, ptr %local.type_name.4688
  %t4702 = load i64, ptr %local.impl_def
  %t4703 = inttoptr i64 %t4702 to ptr
  %t4704 = call ptr @load_ptr(ptr %t4703, i64 3)
  %t4705 = ptrtoint ptr %t4704 to i64
  store i64 %t4705, ptr %local.methods.4689
  %t4706 = call ptr @intrinsic_sb_new()
  %t4707 = ptrtoint ptr %t4706 to i64
  store i64 %t4707, ptr %local.sb.4690
  %t4708 = load i64, ptr %local.sb.4690
  %t4709 = call ptr @intrinsic_string_new(ptr @.str.main.265)
  %t4710 = ptrtoint ptr %t4709 to i64
  %t4711 = inttoptr i64 %t4708 to ptr
  %t4712 = inttoptr i64 %t4710 to ptr
  call void @intrinsic_sb_append(ptr %t4711, ptr %t4712)
  %t4713 = load i64, ptr %local.sb.4690
  %t4714 = load i64, ptr %local.trait_name.4687
  %t4715 = inttoptr i64 %t4713 to ptr
  %t4716 = inttoptr i64 %t4714 to ptr
  call void @intrinsic_sb_append(ptr %t4715, ptr %t4716)
  %t4717 = load i64, ptr %local.sb.4690
  %t4718 = call ptr @intrinsic_string_new(ptr @.str.main.266)
  %t4719 = ptrtoint ptr %t4718 to i64
  %t4720 = inttoptr i64 %t4717 to ptr
  %t4721 = inttoptr i64 %t4719 to ptr
  call void @intrinsic_sb_append(ptr %t4720, ptr %t4721)
  %t4722 = load i64, ptr %local.sb.4690
  %t4723 = load i64, ptr %local.type_name.4688
  %t4724 = inttoptr i64 %t4722 to ptr
  %t4725 = inttoptr i64 %t4723 to ptr
  call void @intrinsic_sb_append(ptr %t4724, ptr %t4725)
  %t4726 = load i64, ptr %local.sb.4690
  %t4727 = call ptr @intrinsic_string_new(ptr @.str.main.267)
  %t4728 = ptrtoint ptr %t4727 to i64
  %t4729 = inttoptr i64 %t4726 to ptr
  %t4730 = inttoptr i64 %t4728 to ptr
  call void @intrinsic_sb_append(ptr %t4729, ptr %t4730)
  store i64 0, ptr %local.i.4691
  br label %loop274
loop274:
  %t4731 = load i64, ptr %local.i.4691
  %t4732 = load i64, ptr %local.methods.4689
  %t4733 = inttoptr i64 %t4732 to ptr
  %t4734 = call i64 @intrinsic_vec_len(ptr %t4733)
  %t4735 = icmp slt i64 %t4731, %t4734
  %t4736 = zext i1 %t4735 to i64
  %t4737 = icmp ne i64 %t4736, 0
  br i1 %t4737, label %body274, label %endloop274
body274:
  %t4738 = load i64, ptr %local.methods.4689
  %t4739 = load i64, ptr %local.i.4691
  %t4740 = inttoptr i64 %t4738 to ptr
  %t4741 = call ptr @intrinsic_vec_get(ptr %t4740, i64 %t4739)
  %t4742 = ptrtoint ptr %t4741 to i64
  store i64 %t4742, ptr %local.method.4692
  %t4743 = load i64, ptr %local.sb.4690
  %t4744 = call ptr @intrinsic_string_new(ptr @.str.main.268)
  %t4745 = ptrtoint ptr %t4744 to i64
  %t4746 = inttoptr i64 %t4743 to ptr
  %t4747 = inttoptr i64 %t4745 to ptr
  call void @intrinsic_sb_append(ptr %t4746, ptr %t4747)
  %t4748 = load i64, ptr %local.method.4692
  %t4749 = call i64 @"format_fn"(i64 %t4748)
  store i64 %t4749, ptr %local.formatted.4693
  %t4750 = load i64, ptr %local.formatted.4693
  %t4751 = call ptr @intrinsic_string_new(ptr @.str.main.269)
  %t4752 = ptrtoint ptr %t4751 to i64
  %t4753 = call ptr @intrinsic_string_new(ptr @.str.main.270)
  %t4754 = ptrtoint ptr %t4753 to i64
  %t4755 = inttoptr i64 %t4750 to ptr
  %t4756 = inttoptr i64 %t4752 to ptr
  %t4757 = inttoptr i64 %t4754 to ptr
  %t4758 = call ptr @intrinsic_string_replace(ptr %t4755, ptr %t4756, ptr %t4757)
  %t4759 = ptrtoint ptr %t4758 to i64
  store i64 %t4759, ptr %local.formatted.4693
  %t4760 = load i64, ptr %local.sb.4690
  %t4761 = load i64, ptr %local.formatted.4693
  %t4762 = inttoptr i64 %t4760 to ptr
  %t4763 = inttoptr i64 %t4761 to ptr
  call void @intrinsic_sb_append(ptr %t4762, ptr %t4763)
  %t4764 = load i64, ptr %local.sb.4690
  %t4765 = call ptr @intrinsic_string_new(ptr @.str.main.271)
  %t4766 = ptrtoint ptr %t4765 to i64
  %t4767 = inttoptr i64 %t4764 to ptr
  %t4768 = inttoptr i64 %t4766 to ptr
  call void @intrinsic_sb_append(ptr %t4767, ptr %t4768)
  %t4769 = load i64, ptr %local.i.4691
  %t4770 = add i64 %t4769, 1
  store i64 %t4770, ptr %local.i.4691
  br label %loop274
endloop274:
  %t4771 = load i64, ptr %local.sb.4690
  %t4772 = call ptr @intrinsic_string_new(ptr @.str.main.272)
  %t4773 = ptrtoint ptr %t4772 to i64
  %t4774 = inttoptr i64 %t4771 to ptr
  %t4775 = inttoptr i64 %t4773 to ptr
  call void @intrinsic_sb_append(ptr %t4774, ptr %t4775)
  %t4776 = load i64, ptr %local.sb.4690
  %t4777 = inttoptr i64 %t4776 to ptr
  %t4778 = call ptr @intrinsic_sb_to_string(ptr %t4777)
  %t4779 = ptrtoint ptr %t4778 to i64
  ret i64 %t4779
}


; String constants
@.str.main.0 = private unnamed_addr constant [7 x i8] c"0.10.0\00"
@.str.main.1 = private unnamed_addr constant [23 x i8] c"sxc - Simplex Compiler\00"
@.str.main.2 = private unnamed_addr constant [1 x i8] c"\00"
@.str.main.3 = private unnamed_addr constant [7 x i8] c"USAGE:\00"
@.str.main.4 = private unnamed_addr constant [29 x i8] c"    sxc [OPTIONS] <FILES...>\00"
@.str.main.5 = private unnamed_addr constant [35 x i8] c"    sxc build [OPTIONS] <FILES...>\00"
@.str.main.6 = private unnamed_addr constant [19 x i8] c"    sxc run <FILE>\00"
@.str.main.7 = private unnamed_addr constant [25 x i8] c"    sxc check <FILES...>\00"
@.str.main.8 = private unnamed_addr constant [53 x i8] c"    sxc repl                        Interactive REPL\00"
@.str.main.9 = private unnamed_addr constant [56 x i8] c"    sxc fmt <FILES...>              Format source files\00"
@.str.main.10 = private unnamed_addr constant [1 x i8] c"\00"
@.str.main.11 = private unnamed_addr constant [9 x i8] c"OPTIONS:\00"
@.str.main.12 = private unnamed_addr constant [37 x i8] c"    -o <file>       Output file name\00"
@.str.main.13 = private unnamed_addr constant [73 x i8] c"    --emit <type>   Output type: llvm-ir (default), asm, obj, exe, dylib\00"
@.str.main.14 = private unnamed_addr constant [58 x i8] c"    -g              Emit DWARF debug symbols for LLDB/GDB\00"
@.str.main.15 = private unnamed_addr constant [35 x i8] c"    -v, --verbose   Verbose output\00"
@.str.main.16 = private unnamed_addr constant [60 x i8] c"    -f, --force     Force recompilation (ignore timestamps)\00"
@.str.main.17 = private unnamed_addr constant [43 x i8] c"    --deps          Show file dependencies\00"
@.str.main.18 = private unnamed_addr constant [53 x i8] c"    --auto-deps     Auto-compile module dependencies\00"
@.str.main.19 = private unnamed_addr constant [35 x i8] c"    -h, --help      Show this help\00"
@.str.main.20 = private unnamed_addr constant [33 x i8] c"    --version       Show version\00"
@.str.main.21 = private unnamed_addr constant [5 x i8] c"sxc \00"
@.str.main.22 = private unnamed_addr constant [8 x i8] c"llvm-ir\00"
@.str.main.23 = private unnamed_addr constant [4 x i8] c"obj\00"
@.str.main.24 = private unnamed_addr constant [4 x i8] c"exe\00"
@.str.main.25 = private unnamed_addr constant [4 x i8] c"asm\00"
@.str.main.26 = private unnamed_addr constant [6 x i8] c"dylib\00"
@.str.main.27 = private unnamed_addr constant [6 x i8] c"build\00"
@.str.main.28 = private unnamed_addr constant [4 x i8] c"run\00"
@.str.main.29 = private unnamed_addr constant [6 x i8] c"check\00"
@.str.main.30 = private unnamed_addr constant [5 x i8] c"repl\00"
@.str.main.31 = private unnamed_addr constant [4 x i8] c"fmt\00"
@.str.main.32 = private unnamed_addr constant [3 x i8] c"-h\00"
@.str.main.33 = private unnamed_addr constant [7 x i8] c"--help\00"
@.str.main.34 = private unnamed_addr constant [10 x i8] c"--version\00"
@.str.main.35 = private unnamed_addr constant [3 x i8] c"-v\00"
@.str.main.36 = private unnamed_addr constant [10 x i8] c"--verbose\00"
@.str.main.37 = private unnamed_addr constant [3 x i8] c"-f\00"
@.str.main.38 = private unnamed_addr constant [8 x i8] c"--force\00"
@.str.main.39 = private unnamed_addr constant [7 x i8] c"--deps\00"
@.str.main.40 = private unnamed_addr constant [12 x i8] c"--auto-deps\00"
@.str.main.41 = private unnamed_addr constant [3 x i8] c"-g\00"
@.str.main.42 = private unnamed_addr constant [3 x i8] c"-o\00"
@.str.main.43 = private unnamed_addr constant [7 x i8] c"--emit\00"
@.str.main.44 = private unnamed_addr constant [20 x i8] c"Unknown emit type: \00"
@.str.main.45 = private unnamed_addr constant [4 x i8] c"mod\00"
@.str.main.46 = private unnamed_addr constant [4 x i8] c".sx\00"
@.str.main.47 = private unnamed_addr constant [4 x i8] c"use\00"
@.str.main.48 = private unnamed_addr constant [4 x i8] c".sx\00"
@.str.main.49 = private unnamed_addr constant [1 x i8] c"\00"
@.str.main.50 = private unnamed_addr constant [11 x i8] c"Compiling \00"
@.str.main.51 = private unnamed_addr constant [25 x i8] c" files with dependencies\00"
@.str.main.52 = private unnamed_addr constant [25 x i8] c"  Compiling dependency: \00"
@.str.main.53 = private unnamed_addr constant [38 x i8] c"Error: Failed to compile dependency: \00"
@.str.main.54 = private unnamed_addr constant [4 x i8] c".ll\00"
@.str.main.55 = private unnamed_addr constant [3 x i8] c".o\00"
@.str.main.56 = private unnamed_addr constant [1 x i8] c"\00"
@.str.main.57 = private unnamed_addr constant [3 x i8] c".s\00"
@.str.main.58 = private unnamed_addr constant [7 x i8] c".dylib\00"
@.str.main.59 = private unnamed_addr constant [24 x i8] c"Skipping (up to date): \00"
@.str.main.60 = private unnamed_addr constant [12 x i8] c"Compiling: \00"
@.str.main.61 = private unnamed_addr constant [22 x i8] c"could not read file: \00"
@.str.main.62 = private unnamed_addr constant [2 x i8] c".\00"
@.str.main.63 = private unnamed_addr constant [4 x i8] c".ll\00"
@.str.main.64 = private unnamed_addr constant [6 x i8] c"  -> \00"
@.str.main.65 = private unnamed_addr constant [3 x i8] c".s\00"
@.str.main.66 = private unnamed_addr constant [32 x i8] c"/usr/local/opt/llvm/bin/llc -o \00"
@.str.main.67 = private unnamed_addr constant [2 x i8] c" \00"
@.str.main.68 = private unnamed_addr constant [18 x i8] c"Error: llc failed\00"
@.str.main.69 = private unnamed_addr constant [6 x i8] c"  -> \00"
@.str.main.70 = private unnamed_addr constant [3 x i8] c".o\00"
@.str.main.71 = private unnamed_addr constant [46 x i8] c"/usr/local/opt/llvm/bin/llc -filetype=obj -o \00"
@.str.main.72 = private unnamed_addr constant [2 x i8] c" \00"
@.str.main.73 = private unnamed_addr constant [18 x i8] c"Error: llc failed\00"
@.str.main.74 = private unnamed_addr constant [6 x i8] c"  -> \00"
@.str.main.75 = private unnamed_addr constant [1 x i8] c"\00"
@.str.main.76 = private unnamed_addr constant [3 x i8] c".o\00"
@.str.main.77 = private unnamed_addr constant [46 x i8] c"/usr/local/opt/llvm/bin/llc -filetype=obj -o \00"
@.str.main.78 = private unnamed_addr constant [2 x i8] c" \00"
@.str.main.79 = private unnamed_addr constant [18 x i8] c"Error: llc failed\00"
@.str.main.80 = private unnamed_addr constant [10 x i8] c"clang -o \00"
@.str.main.81 = private unnamed_addr constant [2 x i8] c" \00"
@.str.main.82 = private unnamed_addr constant [22 x i8] c" standalone_runtime.o\00"
@.str.main.83 = private unnamed_addr constant [22 x i8] c"Error: linking failed\00"
@.str.main.84 = private unnamed_addr constant [6 x i8] c"  -> \00"
@.str.main.85 = private unnamed_addr constant [7 x i8] c".dylib\00"
@.str.main.86 = private unnamed_addr constant [3 x i8] c".o\00"
@.str.main.87 = private unnamed_addr constant [68 x i8] c"/usr/local/opt/llvm/bin/llc -filetype=obj -relocation-model=pic -o \00"
@.str.main.88 = private unnamed_addr constant [2 x i8] c" \00"
@.str.main.89 = private unnamed_addr constant [18 x i8] c"Error: llc failed\00"
@.str.main.90 = private unnamed_addr constant [18 x i8] c"clang -shared -o \00"
@.str.main.91 = private unnamed_addr constant [2 x i8] c" \00"
@.str.main.92 = private unnamed_addr constant [37 x i8] c"Error: linking shared library failed\00"
@.str.main.93 = private unnamed_addr constant [6 x i8] c"  -> \00"
@.str.main.94 = private unnamed_addr constant [32 x i8] c"Error: No input files specified\00"
@.str.main.95 = private unnamed_addr constant [2 x i8] c":\00"
@.str.main.96 = private unnamed_addr constant [20 x i8] c"  (no dependencies)\00"
@.str.main.97 = private unnamed_addr constant [3 x i8] c"  \00"
@.str.main.98 = private unnamed_addr constant [24 x i8] c"Compilation successful.\00"
@.str.main.99 = private unnamed_addr constant [47 x i8] c"Error: 'run' command requires exactly one file\00"
@.str.main.100 = private unnamed_addr constant [1 x i8] c"\00"
@.str.main.101 = private unnamed_addr constant [22 x i8] c"could not read file: \00"
@.str.main.102 = private unnamed_addr constant [5 x i8] c": OK\00"
@.str.main.103 = private unnamed_addr constant [41 x i8] c"Error: No files specified for formatting\00"
@.str.main.104 = private unnamed_addr constant [20 x i8] c"Simplex REPL v0.3.0\00"
@.str.main.105 = private unnamed_addr constant [61 x i8] c"Type expressions to evaluate. Type 'exit' or Ctrl-D to quit.\00"
@.str.main.106 = private unnamed_addr constant [1 x i8] c"\00"
@.str.main.107 = private unnamed_addr constant [4 x i8] c"sx[\00"
@.str.main.108 = private unnamed_addr constant [4 x i8] c"]> \00"
@.str.main.109 = private unnamed_addr constant [1 x i8] c"\00"
@.str.main.110 = private unnamed_addr constant [9 x i8] c"Goodbye!\00"
@.str.main.111 = private unnamed_addr constant [5 x i8] c"exit\00"
@.str.main.112 = private unnamed_addr constant [9 x i8] c"Goodbye!\00"
@.str.main.113 = private unnamed_addr constant [5 x i8] c"quit\00"
@.str.main.114 = private unnamed_addr constant [9 x i8] c"Goodbye!\00"
@.str.main.115 = private unnamed_addr constant [6 x i8] c":help\00"
@.str.main.116 = private unnamed_addr constant [15 x i8] c"REPL Commands:\00"
@.str.main.117 = private unnamed_addr constant [27 x i8] c"  :help     Show this help\00"
@.str.main.118 = private unnamed_addr constant [35 x i8] c"  :vars     Show defined variables\00"
@.str.main.119 = private unnamed_addr constant [32 x i8] c"  :clear    Clear all variables\00"
@.str.main.120 = private unnamed_addr constant [26 x i8] c"  exit      Exit the REPL\00"
@.str.main.121 = private unnamed_addr constant [6 x i8] c":vars\00"
@.str.main.122 = private unnamed_addr constant [22 x i8] c"No variables defined.\00"
@.str.main.123 = private unnamed_addr constant [4 x i8] c" = \00"
@.str.main.124 = private unnamed_addr constant [7 x i8] c":clear\00"
@.str.main.125 = private unnamed_addr constant [19 x i8] c"Variables cleared.\00"
@.str.main.126 = private unnamed_addr constant [5 x i8] c"let \00"
@.str.main.127 = private unnamed_addr constant [22 x i8] c"fn __repl() -> i64 { \00"
@.str.main.128 = private unnamed_addr constant [5 x i8] c" 0 }\00"
@.str.main.129 = private unnamed_addr constant [29 x i8] c"Parse error in let statement\00"
@.str.main.130 = private unnamed_addr constant [1 x i8] c"\00"
@.str.main.131 = private unnamed_addr constant [10 x i8] c"Defined: \00"
@.str.main.132 = private unnamed_addr constant [22 x i8] c"fn __repl() -> i64 { \00"
@.str.main.133 = private unnamed_addr constant [3 x i8] c" }\00"
@.str.main.134 = private unnamed_addr constant [12 x i8] c"Parse error\00"
@.str.main.135 = private unnamed_addr constant [3 x i8] c"= \00"
@.str.main.136 = private unnamed_addr constant [55 x i8] c"Expression parsed OK (evaluation requires compilation)\00"
@.str.main.137 = private unnamed_addr constant [3 x i8] c"= \00"
@.str.main.138 = private unnamed_addr constant [3 x i8] c"= \00"
@.str.main.139 = private unnamed_addr constant [3 x i8] c"= \00"
@.str.main.140 = private unnamed_addr constant [3 x i8] c"= \00"
@.str.main.141 = private unnamed_addr constant [24 x i8] c"Error: Division by zero\00"
@.str.main.142 = private unnamed_addr constant [22 x i8] c"could not read file: \00"
@.str.main.143 = private unnamed_addr constant [12 x i8] c"Formatted: \00"
@.str.main.144 = private unnamed_addr constant [2 x i8] c"\0A\00"
@.str.main.145 = private unnamed_addr constant [1 x i8] c"\00"
@.str.main.146 = private unnamed_addr constant [4 x i8] c"fn \00"
@.str.main.147 = private unnamed_addr constant [2 x i8] c"<\00"
@.str.main.148 = private unnamed_addr constant [3 x i8] c", \00"
@.str.main.149 = private unnamed_addr constant [2 x i8] c">\00"
@.str.main.150 = private unnamed_addr constant [2 x i8] c"(\00"
@.str.main.151 = private unnamed_addr constant [3 x i8] c", \00"
@.str.main.152 = private unnamed_addr constant [3 x i8] c": \00"
@.str.main.153 = private unnamed_addr constant [2 x i8] c")\00"
@.str.main.154 = private unnamed_addr constant [5 x i8] c" -> \00"
@.str.main.155 = private unnamed_addr constant [2 x i8] c" \00"
@.str.main.156 = private unnamed_addr constant [3 x i8] c"{\0A\00"
@.str.main.157 = private unnamed_addr constant [2 x i8] c"\0A\00"
@.str.main.158 = private unnamed_addr constant [2 x i8] c"\0A\00"
@.str.main.159 = private unnamed_addr constant [2 x i8] c"}\00"
@.str.main.160 = private unnamed_addr constant [5 x i8] c"    \00"
@.str.main.161 = private unnamed_addr constant [5 x i8] c"let \00"
@.str.main.162 = private unnamed_addr constant [3 x i8] c": \00"
@.str.main.163 = private unnamed_addr constant [4 x i8] c" = \00"
@.str.main.164 = private unnamed_addr constant [2 x i8] c";\00"
@.str.main.165 = private unnamed_addr constant [2 x i8] c";\00"
@.str.main.166 = private unnamed_addr constant [8 x i8] c"return;\00"
@.str.main.167 = private unnamed_addr constant [8 x i8] c"return \00"
@.str.main.168 = private unnamed_addr constant [2 x i8] c";\00"
@.str.main.169 = private unnamed_addr constant [4 x i8] c" = \00"
@.str.main.170 = private unnamed_addr constant [2 x i8] c";\00"
@.str.main.171 = private unnamed_addr constant [7 x i8] c"break;\00"
@.str.main.172 = private unnamed_addr constant [10 x i8] c"continue;\00"
@.str.main.173 = private unnamed_addr constant [1 x i8] c"\00"
@.str.main.174 = private unnamed_addr constant [1 x i8] c"\00"
@.str.main.175 = private unnamed_addr constant [5 x i8] c"true\00"
@.str.main.176 = private unnamed_addr constant [6 x i8] c"false\00"
@.str.main.177 = private unnamed_addr constant [2 x i8] c"\22\00"
@.str.main.178 = private unnamed_addr constant [2 x i8] c"\22\00"
@.str.main.179 = private unnamed_addr constant [3 x i8] c"::\00"
@.str.main.180 = private unnamed_addr constant [2 x i8] c"(\00"
@.str.main.181 = private unnamed_addr constant [3 x i8] c", \00"
@.str.main.182 = private unnamed_addr constant [2 x i8] c")\00"
@.str.main.183 = private unnamed_addr constant [2 x i8] c" \00"
@.str.main.184 = private unnamed_addr constant [2 x i8] c" \00"
@.str.main.185 = private unnamed_addr constant [2 x i8] c"-\00"
@.str.main.186 = private unnamed_addr constant [2 x i8] c"!\00"
@.str.main.187 = private unnamed_addr constant [4 x i8] c"if \00"
@.str.main.188 = private unnamed_addr constant [2 x i8] c" \00"
@.str.main.189 = private unnamed_addr constant [7 x i8] c" else \00"
@.str.main.190 = private unnamed_addr constant [7 x i8] c"while \00"
@.str.main.191 = private unnamed_addr constant [2 x i8] c" \00"
@.str.main.192 = private unnamed_addr constant [4 x i8] c" { \00"
@.str.main.193 = private unnamed_addr constant [3 x i8] c", \00"
@.str.main.194 = private unnamed_addr constant [3 x i8] c": \00"
@.str.main.195 = private unnamed_addr constant [3 x i8] c" }\00"
@.str.main.196 = private unnamed_addr constant [2 x i8] c".\00"
@.str.main.197 = private unnamed_addr constant [5 x i8] c"for \00"
@.str.main.198 = private unnamed_addr constant [5 x i8] c" in \00"
@.str.main.199 = private unnamed_addr constant [3 x i8] c"..\00"
@.str.main.200 = private unnamed_addr constant [2 x i8] c" \00"
@.str.main.201 = private unnamed_addr constant [7 x i8] c"match \00"
@.str.main.202 = private unnamed_addr constant [4 x i8] c" {\0A\00"
@.str.main.203 = private unnamed_addr constant [2 x i8] c"_\00"
@.str.main.204 = private unnamed_addr constant [5 x i8] c" => \00"
@.str.main.205 = private unnamed_addr constant [3 x i8] c",\0A\00"
@.str.main.206 = private unnamed_addr constant [2 x i8] c"}\00"
@.str.main.207 = private unnamed_addr constant [2 x i8] c".\00"
@.str.main.208 = private unnamed_addr constant [2 x i8] c"(\00"
@.str.main.209 = private unnamed_addr constant [3 x i8] c", \00"
@.str.main.210 = private unnamed_addr constant [2 x i8] c")\00"
@.str.main.211 = private unnamed_addr constant [2 x i8] c"?\00"
@.str.main.212 = private unnamed_addr constant [1 x i8] c"\00"
@.str.main.213 = private unnamed_addr constant [2 x i8] c"+\00"
@.str.main.214 = private unnamed_addr constant [2 x i8] c"-\00"
@.str.main.215 = private unnamed_addr constant [2 x i8] c"*\00"
@.str.main.216 = private unnamed_addr constant [2 x i8] c"/\00"
@.str.main.217 = private unnamed_addr constant [3 x i8] c"==\00"
@.str.main.218 = private unnamed_addr constant [3 x i8] c"!=\00"
@.str.main.219 = private unnamed_addr constant [2 x i8] c"<\00"
@.str.main.220 = private unnamed_addr constant [2 x i8] c">\00"
@.str.main.221 = private unnamed_addr constant [3 x i8] c"<=\00"
@.str.main.222 = private unnamed_addr constant [3 x i8] c">=\00"
@.str.main.223 = private unnamed_addr constant [3 x i8] c"&&\00"
@.str.main.224 = private unnamed_addr constant [3 x i8] c"||\00"
@.str.main.225 = private unnamed_addr constant [2 x i8] c"%\00"
@.str.main.226 = private unnamed_addr constant [2 x i8] c"&\00"
@.str.main.227 = private unnamed_addr constant [2 x i8] c"|\00"
@.str.main.228 = private unnamed_addr constant [2 x i8] c"^\00"
@.str.main.229 = private unnamed_addr constant [3 x i8] c"<<\00"
@.str.main.230 = private unnamed_addr constant [3 x i8] c">>\00"
@.str.main.231 = private unnamed_addr constant [3 x i8] c"??\00"
@.str.main.232 = private unnamed_addr constant [6 x i8] c"enum \00"
@.str.main.233 = private unnamed_addr constant [4 x i8] c" {\0A\00"
@.str.main.234 = private unnamed_addr constant [5 x i8] c"    \00"
@.str.main.235 = private unnamed_addr constant [2 x i8] c",\00"
@.str.main.236 = private unnamed_addr constant [2 x i8] c"\0A\00"
@.str.main.237 = private unnamed_addr constant [2 x i8] c"}\00"
@.str.main.238 = private unnamed_addr constant [8 x i8] c"struct \00"
@.str.main.239 = private unnamed_addr constant [2 x i8] c"<\00"
@.str.main.240 = private unnamed_addr constant [3 x i8] c", \00"
@.str.main.241 = private unnamed_addr constant [2 x i8] c">\00"
@.str.main.242 = private unnamed_addr constant [4 x i8] c" {\0A\00"
@.str.main.243 = private unnamed_addr constant [5 x i8] c"    \00"
@.str.main.244 = private unnamed_addr constant [3 x i8] c": \00"
@.str.main.245 = private unnamed_addr constant [2 x i8] c",\00"
@.str.main.246 = private unnamed_addr constant [2 x i8] c"\0A\00"
@.str.main.247 = private unnamed_addr constant [2 x i8] c"}\00"
@.str.main.248 = private unnamed_addr constant [6 x i8] c"impl \00"
@.str.main.249 = private unnamed_addr constant [4 x i8] c" {\0A\00"
@.str.main.250 = private unnamed_addr constant [5 x i8] c"    \00"
@.str.main.251 = private unnamed_addr constant [2 x i8] c"\0A\00"
@.str.main.252 = private unnamed_addr constant [6 x i8] c"\0A    \00"
@.str.main.253 = private unnamed_addr constant [2 x i8] c"\0A\00"
@.str.main.254 = private unnamed_addr constant [2 x i8] c"}\00"
@.str.main.255 = private unnamed_addr constant [7 x i8] c"trait \00"
@.str.main.256 = private unnamed_addr constant [4 x i8] c" {\0A\00"
@.str.main.257 = private unnamed_addr constant [8 x i8] c"    fn \00"
@.str.main.258 = private unnamed_addr constant [2 x i8] c"(\00"
@.str.main.259 = private unnamed_addr constant [3 x i8] c", \00"
@.str.main.260 = private unnamed_addr constant [3 x i8] c": \00"
@.str.main.261 = private unnamed_addr constant [2 x i8] c")\00"
@.str.main.262 = private unnamed_addr constant [5 x i8] c" -> \00"
@.str.main.263 = private unnamed_addr constant [3 x i8] c";\0A\00"
@.str.main.264 = private unnamed_addr constant [2 x i8] c"}\00"
@.str.main.265 = private unnamed_addr constant [6 x i8] c"impl \00"
@.str.main.266 = private unnamed_addr constant [6 x i8] c" for \00"
@.str.main.267 = private unnamed_addr constant [4 x i8] c" {\0A\00"
@.str.main.268 = private unnamed_addr constant [5 x i8] c"    \00"
@.str.main.269 = private unnamed_addr constant [2 x i8] c"\0A\00"
@.str.main.270 = private unnamed_addr constant [6 x i8] c"\0A    \00"
@.str.main.271 = private unnamed_addr constant [2 x i8] c"\0A\00"
@.str.main.272 = private unnamed_addr constant [2 x i8] c"}\00"
