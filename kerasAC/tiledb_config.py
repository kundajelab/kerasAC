import tiledb
def get_default_config():
    tdb_config=tiledb.Config()
    tdb_config['vfs.s3.region']='us-west-1'
    tdb_config["sm.check_coord_dups"]="false"
    tdb_config["sm.check_coord_oob"]="false"
    tdb_config["sm.check_global_order"]="false"
    tdb_config["sm.num_reader_threads"]="50"
    tdb_config["sm.num_async_threads"]="50"
    tdb_config["vfs.num_threads"]="50"    
    return tdb_config
