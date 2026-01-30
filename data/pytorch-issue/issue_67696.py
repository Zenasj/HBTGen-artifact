for _i in range(200):
            for scheduler in schedulers:
                if isinstance(scheduler, ReduceLROnPlateau):
                    val_loss = fn()
                    scheduler.step(val_loss)
                else:
                    scheduler.step()
            optimizer.step(fn)