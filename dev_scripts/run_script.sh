#!/bin/sh
poetry run python src/train.py dataset=YBCO13_fold0_paper
poetry run python src/train.py dataset=YBCO13_fold1_paper
poetry run python src/train.py dataset=YBCO13_fold2_paper
poetry run python src/train.py dataset=YBCO13_fold3_paper
poetry run python src/train.py dataset=YBCO13_fold4_paper
poetry run python src/train.py dataset=YBCO13_fold5_paper
poetry run python src/train.py dataset=YBCO13_fold6_paper
poetry run python src/train.py dataset=YBCO13_fold7_paper
poetry run python src/train.py dataset=YBCO13_fold8_paper
poetry run python src/train.py dataset=YBCO13_fold9_paper
poetry run python src/train.py dataset=YBCO13_fold10_paper
poetry run python src/train.py dataset=YBCO13_fold11_paper
poetry run python src/train.py dataset=YBCO13_fold12_paper
poetry run python src/train.py dataset=YBCO13_fold13_paper
poetry run python src/train.py dataset=YBCO13_fold14_paper
poetry run python src/train.py dataset=YBCO13_fold15_paper
poetry run python src/train.py dataset=YBCO13_fold16_paper
poetry run python src/train.py dataset=YBCO13_fold17_paper
poetry run python src/train.py dataset=YBCO13_fold18_paper
poetry run python src/train.py dataset=YBCO13_fold19_paper
poetry run python src/train.py dataset=ICSG3D_paper
poetry run python src/train.py dataset=lim_l6_paper

