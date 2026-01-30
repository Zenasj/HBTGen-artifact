def run(self, args, env):
        "Executes cmake with arguments and an environment."

        command = [self._cmake_command] + args
        print()
        print(' '.join(command))
        print()
        #check_call(command, cwd=self.build_dir, env=env)