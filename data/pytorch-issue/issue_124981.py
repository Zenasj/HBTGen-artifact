out[0]._cdata in  [ref()._cdata for ab in non_cudagraph_inps_storage_refs] # False 
out[0].data_ptr() in  [ref().data_ptr() for ab in non_cudagraph_inps_storage_refs] # True