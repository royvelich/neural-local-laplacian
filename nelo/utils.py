# Stub for NeLo's utils.py.
# These functions are only called inside the `if global_config.use_ref_mesh` branch,
# which is never entered when ref_mesh=None (our inference path).
# Raise clearly if anything somehow calls them.

def _not_available(name):
    def _fn(*args, **kwargs):
        raise NotImplementedError(
            f"utils.{name} is a stub — it should not be called during point-cloud inference."
        )
    _fn.__name__ = name
    return _fn

get_GT_L_and_M                      = _not_available("get_GT_L_and_M")
get_GT_eigens_of_mesh                = _not_available("get_GT_eigens_of_mesh")
check_closeness                      = _not_available("check_closeness")
add_noise_to_vertices_position       = _not_available("add_noise_to_vertices_position")
check_if_exist_multi_edge            = _not_available("check_if_exist_multi_edge")
convert_dense_tensor_to_sparse_numpy = _not_available("convert_dense_tensor_to_sparse_numpy")
convert_sparse_numpy_to_dense_tensor = _not_available("convert_sparse_numpy_to_dense_tensor")
relative_mass_matrix                 = _not_available("relative_mass_matrix")
