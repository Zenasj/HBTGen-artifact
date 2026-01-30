for name in names_to_remove:
            if name in V.kernel.args.inplace_buffers:
                buf = V.kernel.args.inplace_buffers[name]
                if buf == "REMOVED": 
                    continue
                remove = all(n in names_to_remove for n in buf.other_names)
                if remove:
                    self.remove_inplace_buffer(name)
                V.graph.inplaced_to_remove.add(name)
            else:
                self.remove_buffer(name)