import tempfile

from pygradflow.params import Params, PenaltyUpdate


def test_roundtrip():
    params = Params(rho=1.0, penalty_update=PenaltyUpdate.Constant)

    with tempfile.TemporaryDirectory() as tmp:
        filename = tmp + "/params.yml"
        params.write(filename)
        read_params = Params.read(filename)
        assert params == read_params
