from pathlib import Path


def exp_complete(folder):
    if folder.exists():
        dirs = [x for x in folder.iterdir() if x.is_dir()]
        if len(dirs) == 16:
            return True
    return False


def amine_complete(amine_path):
    if amine_path.exists():
        files_in_amine = [x for x in amine_path.iterdir()]
        # If 1k-5k + cv_stat files exist dont run the amine
        if len(files_in_amine) == 6:
            return True
    return False


if __name__ == "__main__":
    incomplete = []
    result_path = Path("./results")
    for exp_path in result_path.iterdir():
        print(f"Scanning {exp_path.name}")
        if exp_path.name != "without_1k_al" and not exp_complete(exp_path):
            for amine_path in exp_path.iterdir():
                if amine_path.is_dir():
                    if not amine_complete(amine_path):
                        exp_num = int(exp_path.name.split("_")[1])
                        incomplete.append(exp_num)
                        print(f"Amine incomplete {amine_path.name}")
    print(incomplete)
