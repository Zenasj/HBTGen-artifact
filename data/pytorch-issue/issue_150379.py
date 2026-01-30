for key, value in counters["aten_mm_info"].items():
            name, m, n, k = key.split("_")

counters["aten_mm_info"][f"aten._int_mm_{m}_{n}_{k}"] += 1