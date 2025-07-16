from ..files import from_files_directory


class FitOptions:
    data_file = from_files_directory("data/rp-data.pickle")
    result_file = from_files_directory("results/least-squares-fit.pickle")
    boostrap_file = from_files_directory("results/bootstrap-fit.pickle")

    rp_file_id = "1kBrPl0pdAO8gjOf_NrTgAPseFtqQA27fdfEbMBBeAhs"

    rp_sheet_ids = {
        "data_1_9": "1682088244",
        "data_10_49": "1691041080",
        "data_50_74": "161092121",
        "legacy": "2047819558",
        "pokedex": "513440248",
        "main_skill": "1395455629",
    }
