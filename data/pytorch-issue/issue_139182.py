assert options.num_stages != 0, ("Triton AMD backend pipeliner has been updated. "
                                 "We used to trigger software pipelining with "
                                 "num_stages == 0. Now it will not happen anymore; "
                                 "please update to use num_stages == 2 for "
                                 "equivalent behavior in the past.")