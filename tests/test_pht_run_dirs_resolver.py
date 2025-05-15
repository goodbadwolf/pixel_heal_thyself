import os
import tempfile
import pytest
from pht.hydra.plugins.pht_run_dirs_resolver import (
    pht_run_dirs_resolver,
    register_pht_run_dirs_resolver,
    reset_pht_run_dirs_cache,
)


@pytest.fixture(autouse=True)
def register_and_reset_resolver():
    register_pht_run_dirs_resolver()
    reset_pht_run_dirs_cache()


@pytest.fixture
def temp_cwd(monkeypatch):
    with tempfile.TemporaryDirectory() as tmpdir:
        monkeypatch.setattr(os, "getcwd", lambda: tmpdir)
        yield tmpdir


def test_single_run_creates_dir(temp_cwd):
    base_pattern = "modelA_p32_n100_r2"
    output_dir = pht_run_dirs_resolver(
        cfg_base_pattern=base_pattern,
        cfg_job_subdir=None,
        cfg_run_num=-1,
        cfg_is_multirun=False,
    )
    assert os.path.exists(output_dir)
    assert base_pattern in output_dir
    assert output_dir.endswith("run000")
    # Directory should be empty
    assert os.listdir(output_dir) == []


def test_single_run_with_explicit_run_num(temp_cwd):
    base_pattern = "modelB_p16_n50_r1"
    output_dir = pht_run_dirs_resolver(
        cfg_base_pattern=base_pattern,
        cfg_job_subdir=None,
        cfg_run_num=5,
        cfg_is_multirun=False,
    )
    assert os.path.exists(output_dir)
    assert output_dir.endswith("run005")


def test_multirun_creates_dir(temp_cwd):
    base_pattern = "modelC_p8_n25_r3"
    output_dir = pht_run_dirs_resolver(
        cfg_base_pattern=base_pattern,
        cfg_job_subdir=None,
        cfg_run_num=-1,
        cfg_is_multirun=True,
    )
    assert os.path.exists(output_dir)
    assert "multiruns" in output_dir
    assert output_dir.endswith("run000")


def test_multirun_with_cfg_job_subdir(temp_cwd):
    base_pattern = "modelD_p64_n200_r4"
    cfg_job_subdir = "job42"
    output_dir = pht_run_dirs_resolver(
        cfg_base_pattern=base_pattern,
        cfg_job_subdir=cfg_job_subdir,
        cfg_run_num=-1,
        cfg_is_multirun=True,
    )
    assert os.path.exists(os.path.dirname(output_dir))
    assert output_dir.endswith(f"run000/{cfg_job_subdir}")
    # The parent directory should exist
    assert os.path.exists(os.path.dirname(output_dir))


def test_cache_behavior(temp_cwd):
    base_pattern = "modelE_p32_n100_r2"
    output_dir1 = pht_run_dirs_resolver(
        cfg_base_pattern=base_pattern,
        cfg_job_subdir=None,
        cfg_run_num=-1,
        cfg_is_multirun=False,
    )
    output_dir2 = pht_run_dirs_resolver(
        cfg_base_pattern=base_pattern,
        cfg_job_subdir=None,
        cfg_run_num=-1,
        cfg_is_multirun=False,
    )
    assert output_dir1 == output_dir2
    # Changing cfg_job_subdir should not affect cache for single run
    output_dir3 = pht_run_dirs_resolver(
        cfg_base_pattern=base_pattern,
        cfg_job_subdir="jobX",
        cfg_run_num=-1,
        cfg_is_multirun=False,
    )
    assert output_dir3 == output_dir1


def test_run_num_as_string(temp_cwd):
    base_pattern = "modelF_p32_n100_r2"
    output_dir = pht_run_dirs_resolver(
        cfg_base_pattern=base_pattern,
        cfg_job_subdir=None,
        cfg_run_num="7",
        cfg_is_multirun=False,
    )
    assert output_dir.endswith("run007")


def test_cfg_job_subdir_none_string(temp_cwd):
    base_pattern = "modelG_p32_n100_r2"
    output_dir = pht_run_dirs_resolver(
        cfg_base_pattern=base_pattern,
        cfg_job_subdir="None",
        cfg_run_num=-1,
        cfg_is_multirun=False,
    )
    assert "None" not in output_dir


def test_existing_directory(temp_cwd):
    base_pattern = "modelH_p32_n100_r2"
    base_dir = os.path.join(temp_cwd, "runs", base_pattern)
    os.makedirs(os.path.join(base_dir, "run000"), exist_ok=True)
    output_dir = pht_run_dirs_resolver(
        cfg_base_pattern=base_pattern,
        cfg_job_subdir=None,
        cfg_run_num=-1,
        cfg_is_multirun=False,
    )
    assert output_dir.endswith("run001")


def test_multiple_runs_increment(temp_cwd):
    base_pattern = "modelI_p32_n100_r2"
    base_dir = os.path.join(temp_cwd, "runs", base_pattern)
    os.makedirs(os.path.join(base_dir, "run000"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "run001"), exist_ok=True)
    output_dir = pht_run_dirs_resolver(
        cfg_base_pattern=base_pattern,
        cfg_job_subdir=None,
        cfg_run_num=-1,
        cfg_is_multirun=False,
    )
    assert output_dir.endswith("run002")


@pytest.mark.parametrize(
    "is_multirun,expected_dir",
    [
        (False, "runs"),
        (True, "multiruns"),
    ],
)
def test_dir_type_switch(temp_cwd, is_multirun, expected_dir):
    base_pattern = "modelJ_p32_n100_r2"
    output_dir = pht_run_dirs_resolver(
        cfg_base_pattern=base_pattern,
        cfg_job_subdir=None,
        cfg_run_num=-1,
        cfg_is_multirun=is_multirun,
    )
    assert expected_dir in output_dir


def test_cfg_job_subdir_with_special_chars(temp_cwd):
    base_pattern = "modelK_p32_n100_r2"
    cfg_job_subdir = "job-42_!@#"
    output_dir = pht_run_dirs_resolver(
        cfg_base_pattern=base_pattern,
        cfg_job_subdir=cfg_job_subdir,
        cfg_run_num=-1,
        cfg_is_multirun=True,
    )
    assert output_dir.endswith(f"run000/{cfg_job_subdir}")


def test_run_num_zero(temp_cwd):
    base_pattern = "modelL_p32_n100_r2"
    output_dir = pht_run_dirs_resolver(
        cfg_base_pattern=base_pattern,
        cfg_job_subdir=None,
        cfg_run_num=0,
        cfg_is_multirun=False,
    )
    assert output_dir.endswith("run000")


def test_invalid_run_num(temp_cwd):
    base_pattern = "modelM_p32_n100_r2"
    output_dir = pht_run_dirs_resolver(
        cfg_base_pattern=base_pattern,
        cfg_job_subdir=None,
        cfg_run_num="notanumber",
        cfg_is_multirun=False,
    )
    assert output_dir.endswith("run000")


def test_dir_type_argument_ignored(temp_cwd):
    base_pattern = "modelN_p32_n100_r2"
    output_dir = pht_run_dirs_resolver(
        cfg_base_pattern=base_pattern,
        cfg_job_subdir=None,
        cfg_run_num=-1,
        cfg_is_multirun=False,
    )
    assert output_dir.endswith("run000")


def test_multiple_cfg_job_subdirs_multirun(temp_cwd):
    base_pattern = "modelO_p32_n100_r2"
    cfg_job_subdirs = ["job1", "job2", "job3"]
    dirs = [
        pht_run_dirs_resolver(
            cfg_base_pattern=base_pattern,
            cfg_job_subdir=jid,
            cfg_run_num=-1,
            cfg_is_multirun=True,
        )
        for jid in cfg_job_subdirs
    ]
    for jid, d in zip(cfg_job_subdirs, dirs):
        assert d.endswith(f"run000/{jid}")
    # All share the same parent run dir
    parent = os.path.dirname(dirs[0])
    for d in dirs:
        assert os.path.dirname(d) == parent


def test_cache_reset_between_tests_without_delete(temp_cwd):
    base_pattern = "modelP_p32_n100_r2"
    output_dir1 = pht_run_dirs_resolver(
        cfg_base_pattern=base_pattern,
        cfg_job_subdir=None,
        cfg_run_num=-1,
        cfg_is_multirun=False,
    )
    assert output_dir1.endswith("run000")
    reset_pht_run_dirs_cache()
    output_dir2 = pht_run_dirs_resolver(
        cfg_base_pattern=base_pattern,
        cfg_job_subdir=None,
        cfg_run_num=-1,
        cfg_is_multirun=False,
    )
    assert output_dir2.endswith("run001")
    assert output_dir1 != output_dir2


def test_cache_reset_between_tests_with_delete(temp_cwd):
    base_pattern = "modelP_p32_n100_r2"
    output_dir1 = pht_run_dirs_resolver(
        cfg_base_pattern=base_pattern,
        cfg_job_subdir=None,
        cfg_run_num=-1,
        cfg_is_multirun=False,
    )
    assert output_dir1.endswith("run000")
    reset_pht_run_dirs_cache()
    os.rmdir(output_dir1)
    output_dir2 = pht_run_dirs_resolver(
        cfg_base_pattern=base_pattern,
        cfg_job_subdir=None,
        cfg_run_num=-1,
        cfg_is_multirun=False,
    )
    assert output_dir2.endswith("run000")
    assert output_dir1 == output_dir2
