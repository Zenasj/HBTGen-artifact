with open(options.TEST_ONLY_op_registration_allowlist_yaml_path, "r") as f:
        op_registration_allowlist = yaml.safe_load(f)