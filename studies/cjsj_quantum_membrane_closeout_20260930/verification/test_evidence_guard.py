from pathlib import Path
import hashlib,json,sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"source"))
from manuscript_evidence import verify_evidence,EvidenceMismatch

def fixture(tmp_path):
    data=tmp_path/"data.csv";data.write_text("x,y\n1,2\n")
    lock=tmp_path/"lock.json"
    lock.write_text(json.dumps({"schema":1,"files":{"data.csv":hashlib.sha256(data.read_bytes()).hexdigest()}}))
    return data,lock

def test_current_archive():
    assert verify_evidence(Path(__file__).resolve().parents[1])["locked_inputs"]>70

def test_unchanged(tmp_path):
    data,lock=fixture(tmp_path);assert verify_evidence(tmp_path,lock)["status"]=="PASS"

def test_mutation_rejected(tmp_path):
    data,lock=fixture(tmp_path);data.write_text("x,y\n1,99\n")
    with pytest.raises(EvidenceMismatch,match="changed"):
        verify_evidence(tmp_path,lock)

def test_missing_evidence_rejected(tmp_path):
    data,lock=fixture(tmp_path);data.unlink()
    with pytest.raises(EvidenceMismatch,match="missing"):
        verify_evidence(tmp_path,lock)

def test_missing_lock_rejected(tmp_path):
    with pytest.raises(EvidenceMismatch,match="Missing"):
        verify_evidence(tmp_path)

def test_malformed_lock_rejected(tmp_path):
    data,lock=fixture(tmp_path);lock.write_text("invalid json")
    with pytest.raises(EvidenceMismatch,match="Malformed"):
        verify_evidence(tmp_path,lock)

def test_escape_rejected(tmp_path):
    data,lock=fixture(tmp_path);lock.write_text(json.dumps({"schema":1,"files":{"../outside":"00"}}))
    with pytest.raises(EvidenceMismatch,match="Unsafe"):
        verify_evidence(tmp_path,lock)
