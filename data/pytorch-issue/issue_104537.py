TestEnvironment.def_flag("TEST_WITH_TORCHINDUCTOR", env_var="PYTORCH_TEST_WITH_INDUCTOR")
# can track implication relationships to avoid adding unnecessary flags to the repro
TestEnvironment.def_flag(
    "TEST_WITH_TORCHDYNAMO",
    env_var="PYTORCH_TEST_WITH_DYNAMO",
    implied_by_fn=lambda: TEST_WITH_TORCHINDUCTOR or TEST_WITH_AOT_EAGER)
# can use include_in_repro=False to keep the flag from appearing in the repro command
TestEnvironment.def_flag(
    "DISABLE_RUNNING_SCRIPT_CHK", env_var="PYTORCH_DISABLE_RUNNING_SCRIPT_CHK", include_in_repro=False)
# the default default value is False, but this can be changed
TestEnvironment.def_flag(
    "PRINT_REPRO_ON_FAILURE", env_var="PYTORCH_PRINT_REPRO_ON_FAILURE", default=(not IS_FBCODE), include_in_repro=False)

IS_CI = bool(os.getenv('CI'))