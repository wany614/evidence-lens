"""Clear only Blender's saved file-browser directory in an authorized public copy."""
import hashlib
from pathlib import Path
import struct


def require(ok,reason):
    if not ok: raise ValueError(reason)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def decode(path):
    import pyarrow as pa
    with path.open('rb') as f:magic=f.read(4)
    if magic==b'BLEN':return path.read_bytes()
    require(magic==b'\x28\xb5\x2f\xfd','Unsupported Blender compression')
    with pa.input_stream(str(path),compression='zstd') as stream:
        return stream.read()


def blocks(data):
    require(data.startswith(b'BLENDER-v405'),'Only the registered 64-bit little-endian Blender 4.5 binary format is supported')
    off=12
    while off+24<=len(data):
        code,size,unused,sdna,count=struct.unpack_from('<4siQii',data,off)
        pos=off+24
        require(size>=0 and pos+size<=len(data),'Bad native block size')
        yield code,pos,size,sdna,count
        off=pos+size
        if code==b'ENDB':
            require(off==len(data),'Unparsed bytes after final Blender block')
            return
    raise ValueError('Missing final Blender block')


def filebrowser_directory(dna):
    require(dna[:4]==b'SDNA','Missing Blender DNA')
    p=4
    lists=[]
    for tag in (b'NAME',b'TYPE'):
        require(dna[p:p+4]==tag,'Unexpected DNA section')
        n,=struct.unpack_from('<I',dna,p+4)
        p+=8
        strings=[]
        for i in range(n):
            end=dna.index(0,p);strings.append(dna[p:end].decode());p=end+1
        lists.append(strings)
        p=(p+3)&~3
    names,types=lists
    require(dna[p:p+4]==b'TLEN','Missing field sizes')
    sizes=struct.unpack_from('<'+'H'*len(types),dna,p+4)
    p=(p+4+len(types)*2+3)&~3
    require(dna[p:p+4]==b'STRC','Missing structures')
    total,=struct.unpack_from('<I',dna,p+4);p+=8
    for index in range(total):
        typ,n=struct.unpack_from('<HH',dna,p);p+=4
        fields=[]
        for _ in range(n):
            t,name=struct.unpack_from('<HH',dna,p)
            p+=4;fields.append((types[t],names[name]))
        if types[typ]=='FileSelectParams':
            require(fields[:2]==[('char','title[96]'),('char','dir[1090]')] and sizes[typ]==2088,
                    'Native file-browser directory layout changed')
            return index,96,1090,2088
    raise ValueError('FileSelectParams absent')


def portable_copy(source,destination):
    native=decode(source)
    layout=list(blocks(native))
    catalog=[native[p:p+n] for k,p,n,i,c in layout if k==b'DNA1']
    require(len(catalog)==1,'Unexpected native DNA count')
    type_id,field_at,field_size,record_size=filebrowser_directory(catalog[0])
    value=bytearray(native)
    intervals=[]
    for kind,p,n,typ,count in layout:
        if typ!=type_id or kind!=b'DATA':continue
        require(n==record_size*count,'Unexpected file-browser record size')
        for i in range(count):
            start=p+i*record_size+field_at
            stop=start+field_size
            value[start:stop]=bytes(field_size)
            intervals.append((start,stop))
    require(intervals,'No native file-browser directory found')
    cursor=0
    unchanged=hashlib.sha256()
    for start,stop in intervals:
        require(value[cursor:start]==native[cursor:start] and value[start:stop]==bytes(field_size),'Unexpected scientific-byte change')
        unchanged.update(native[cursor:start]);cursor=stop
    require(value[cursor:]==native[cursor:],'Unexpected final block change')
    unchanged.update(native[cursor:])
    # A complete clear also removes stale bytes after the original NUL terminator.
    require(b'/Users/' not in value and b'/home/' not in value and b'Library/CloudStorage' not in value,'Another personal directory remains')
    destination.parent.mkdir(parents=True,exist_ok=True)
    destination.write_bytes(value)
    return {'schema':'el-stage3-public-blend-projection-1','source_container_sha256':digest(source.read_bytes()),
            'source_decoded_sha256':digest(native),'portable_native_sha256':digest(value),'decoded_bytes':len(native),
            'field':'FileSelectParams.dir[1090]','cleared_fields':len(intervals),
            'unmodified_outside_fields_sha256':unchanged.hexdigest(),'unchanged_bytes':len(native)-sum(b-a for a,b in intervals),
            'changed_byte_count':sum(x!=y for x,y in zip(native,value)),'output_format':'uncompressed native Blender file',
            'scope':'All bytes outside the verified file-browser directory fields are identical, including scene, meshes, materials and cameras.'}
