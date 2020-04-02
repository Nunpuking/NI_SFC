TOPO_NAME='internet2'
DATA_DIR='../data/'
TOPO_FILE='inet2'
MIDDLEBOX_FILE='middlebox-spec'
SFCTYPE_FILE='sfctypes'
REQUEST_FILE='20190530-requests.csv'
NODEINFO_FILE='20190530-nodeinfo.csv'
ROUTEINFO_FILE='20190530-routeinfo.csv'

python3 preproc_label.py\
    --topology_name=$TOPO_NAME --data_dir=$DATA_DIR --topo_file=$TOPO_FILE\
    --middlebox_file=$MIDDLEBOX_FILE --sfctypes_file=$SFCTYPE_FILE\
    --request_file=$REQUEST_FILE --nodeinfo_file=$NODEINFO_FILE\
    --routeinfo_file=$ROUTEINFO_FILE
