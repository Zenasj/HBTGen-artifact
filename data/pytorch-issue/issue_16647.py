def _process_next_batch(self, batch):
        self.rcvd_idx += 1
        self._put_indices()
        if isinstance(batch, ExceptionWrapper):
            raise batch.exc_type(batch.exc_msg)
        return batch

def _process_next_batch(self, batch):
        self.rcvd_idx += 1
        self._put_indices()
        if isinstance(batch, ExceptionWrapper):
            # help with multiline KeyError msg to be readable
            if batch.exc_type == KeyError and "\n" in batch.exc_msg:
                raise Exception("KeyError:" + batch.exc_msg)
            else:
                raise batch.exc_type(batch.exc_msg)
        return batch