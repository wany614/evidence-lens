import hashlib
import io
import struct
import sys
import zipfile
from pathlib import Path

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from t0_dtu_archive import ExactArchive, MetadataError
from t0_dtu_convert import METHODS, RECIPE, TARGETS, REFERENCE_SOFTWARE, authorise, verify_reference_software
from t0_dtu_roster import components, matlab_ids


class InMemoryRange:
    def __init__(self,data):
        self.data = data
        self.requests = []

    def get(self,url,*,span,limit,total=None,etag=None):
        a,b = span
        assert 0 <= a <= b < len(self.data) and limit == b-a+1
        self.requests.append((a,b+1))
        return self.data[a:b+1], {"content-range":f"bytes {a}-{b}/{len(self.data)}","etag":'"fixture"',
                                  "last-modified":"Mon, 01 Jan 2024 00:00:00 GMT"}


@pytest.mark.parametrize("method",[zipfile.ZIP_STORED,zipfile.ZIP_DEFLATED])
def test_directory_and_one_source_never_fetch_neighbouring_photograph(method):
    f=io.BytesIO()
    with zipfile.ZipFile(f,"w",compression=method) as z:
        for n in ("allowed","closed"):
            z.writestr(n, (n + ".").encode()*1300)
    raw=f.getvalue()
    http=InMemoryRange(raw)
    az=ExactArchive(http,"https://fixture")
    identity,entries,central=az.directory()
    assert hashlib.sha256(central).hexdigest()==identity["directory_sha256"]
    closed=entries["closed"]
    body=closed["offset"]+30+len("closed")
    assert all(start>=identity["directory_offset"] or end<=4 for start,end in http.requests)
    assert az.member(entries["allowed"]) == b"allowed."*1300
    assert all(end<=body or start>=body+closed["compressed"] for start,end in http.requests)
    with pytest.raises(MetadataError,match="different member"):
        az.member({**entries["allowed"],"name":"closed"})


def test_active_author_list_and_transitive_rotation_uncertainty():
    assert matlab_ids(b"%UsedSets=[99];\nUsedSets=[1:3 5];") == {1,2,3,5}
    groups=components([118,119,120,121,122,123,124,125],[[118,119,120,121],[122,123,124,125]],
                      {118:"stone statue",123:"stone statue"})
    assert set(groups.values())=={118}


def example_pin(rid,member,letter):
    return {"run_id":f"20261124T010101111111Z-{rid:016x}","manifest_sha256":letter*64,"member":"payload/"+member,"sha256":letter*64}


def fixture():
    rpin=example_pin(1,"roster.json","a"); fpin=example_pin(2,"freeze.json","b");opin=example_pin(3,"opening.json","c")
    roster={"schema":"el-t0-dtu-roster-1","object_ids":["set2"],"objects":[{"object_id":"set2","scan_id":2,"physical_family_id":"subject2"}],
            "conditions":["M2","M4"],"lighting_id":3}
    fr={"schema":"el-t0-freeze-1","run_id":fpin["run_id"],"sources":{"roster":rpin},"roster_sha256":rpin["sha256"],"stop_step":5000,
        "recipe_sha256":RECIPE,"models":dict.fromkeys(METHODS,{}),"conditions":["M2","M4"],"target_ids":list(TARGETS),"object_ids":["set2"],
        "target_reference_permission":"closed","expected_scores":48}
    cfg={"role":"sources","condition":"M2","object_id":"set2","roster":rpin,"freeze":fpin,"opening":None}
    marker={"schema":"el-t0-reference-opening-1","run_id":opin["run_id"],"freeze":fpin,"roster":rpin,"roster_sha256":rpin["sha256"],
            "object_ids":["set2"],"conditions":["M2","M4"],"target_ids":list(TARGETS),"permitted_reader":"independent_evaluator",
            "predictions_and_risks_sealed":True,"scheduled_entries":48,"scored_entries":0,"sealed_matrix_sha256":"d"*64,
            "terminal_cells":[{"object_id":"set2","condition":c,"status":"failed","reason":"fixture","disposition":example_pin(i,"failed.json","f")}
                              for i,c in enumerate(("M2","M4"),4)]}
    marker["software_lock"] = example_pin(6,"software-lock.json","a")
    return roster,cfg,fr,opin,marker


def test_source_endpoints_and_collective_reference_gate_are_independent():
    roster,cfg,fr,opin,marker=fixture()
    assert authorise(roster,cfg,fr,None)[1]==("s00","s03")
    assert authorise(roster,{**cfg,"condition":"M4"},fr,None)[1]==("s00","s01","s02","s03")
    refs={**cfg,"role":"references","condition":None,"opening":opin,"software_lock":marker["software_lock"]}
    with pytest.raises(MetadataError): authorise(roster,refs,fr,None)
    with pytest.raises(MetadataError): authorise(roster,refs,fr,{**marker,"terminal_cells":marker["terminal_cells"][:1]})
    with pytest.raises(MetadataError): authorise(roster,refs,fr,{**marker,"roster_sha256":"e"*64})
    with pytest.raises(MetadataError): authorise(roster,{**cfg,"object_id":"set42"},fr,None)
    assert authorise(roster,refs,fr,marker)[1]==TARGETS


def test_small_zip64_directory_and_high_local_offset_resolve_without_overlap():
    buf=io.BytesIO()
    with zipfile.ZipFile(buf,"w",compression=zipfile.ZIP_STORED) as z:
        z.writestr("one",b"metadata-boundary-test")
    old=buf.getvalue()
    eocd=struct.unpack("<4s4H2IH",old[-22:])
    directory_position=eocd[6]
    central=list(struct.unpack("<4s6H3I5H2I",old[directory_position:directory_position+46]))
    name=old[directory_position+46:directory_position+46+central[10]]
    true_unpacked,true_compressed,true_offset=central[9],central[8],central[16]
    extra=struct.pack("<HHQQQ",1,24,true_unpacked,true_compressed,true_offset)
    central[9]=central[8]=central[16]=0xffffffff
    central[11]=len(extra)
    cd=struct.pack("<4s6H3I5H2I",*central)+name+extra
    zip64_position=directory_position+len(cd)
    z64=struct.pack("<4sQ2H2I4Q",b"PK\x06\x06",44,45,45,0,0,1,1,len(cd),directory_position)
    loc=struct.pack("<4sIQI",b"PK\x06\x07",0,zip64_position,1)
    end=struct.pack("<4s4H2IH",b"PK\x05\x06",0,0,65535,65535,0xffffffff,0xffffffff,0)
    http=InMemoryRange(old[:directory_position]+cd+z64+loc+end)
    archive=ExactArchive(http,"https://fixture")
    meta,rows,_=archive.directory()
    assert meta["entries"]==1
    assert rows["one"]["offset"]==0 and rows["one"]["bytes"]==true_unpacked
    assert all(start>=directory_position or end<=4 for start,end in http.requests)
    assert archive.member(rows["one"])==b"metadata-boundary-test"



def test_reference_software_lock_binds_primary_freeze_and_every_runtime_member(tmp_path):
    _,cfg,_,_,marker=fixture()
    files={}
    for n in REFERENCE_SOFTWARE:
        file=tmp_path/n
        file.parent.mkdir(parents=True,exist_ok=True)
        file.write_bytes(("fixture for "+n).encode())
        files[n]=hashlib.sha256(file.read_bytes()).hexdigest()
    lock={"schema":"el-t0-opening-software-1","sources":{"freeze":cfg["freeze"]},"files":files}
    verify_reference_software(lock,cfg["freeze"],tmp_path)
    with pytest.raises(MetadataError): verify_reference_software(lock,example_pin(4,"freeze.json","e"),tmp_path)
    for n in REFERENCE_SOFTWARE:
        file=tmp_path/n; old=file.read_bytes();file.write_bytes(old+b"changed")
        with pytest.raises(MetadataError,match="unlocked reference implementation"): verify_reference_software(lock,cfg["freeze"],tmp_path)
        file.write_bytes(old)
