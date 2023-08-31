#!/bin/sh
poetry run python src/train.py dataset=YBCO13_fold0_paper trainer.max_epochs=1
poetry run python src/train.py dataset=YBCO13_fold1_paper trainer.max_epochs=1
poetry run python src/train.py dataset=YBCO13_fold2_paper trainer.max_epochs=1
poetry run python src/train.py dataset=YBCO13_fold3_paper trainer.max_epochs=1
poetry run python src/train.py dataset=YBCO13_fold4_paper trainer.max_epochs=1
poetry run python src/train.py dataset=YBCO13_fold5_paper trainer.max_epochs=1
poetry run python src/train.py dataset=YBCO13_fold6_paper trainer.max_epochs=1
poetry run python src/train.py dataset=YBCO13_fold7_paper trainer.max_epochs=1
poetry run python src/train.py dataset=YBCO13_fold8_paper trainer.max_epochs=1
poetry run python src/train.py dataset=YBCO13_fold9_paper trainer.max_epochs=1
poetry run python src/train.py dataset=YBCO13_fold10_paper trainer.max_epochs=1
poetry run python src/train.py dataset=YBCO13_fold11_paper trainer.max_epochs=1
poetry run python src/train.py dataset=YBCO13_fold12_paper trainer.max_epochs=1
poetry run python src/train.py dataset=YBCO13_fold13_paper trainer.max_epochs=1
poetry run python src/train.py dataset=YBCO13_fold14_paper trainer.max_epochs=1
poetry run python src/train.py dataset=YBCO13_fold15_paper trainer.max_epochs=1
poetry run python src/train.py dataset=YBCO13_fold16_paper trainer.max_epochs=1
poetry run python src/train.py dataset=YBCO13_fold17_paper trainer.max_epochs=1
poetry run python src/train.py dataset=YBCO13_fold18_paper trainer.max_epochs=1
poetry run python src/train.py dataset=YBCO13_fold19_paper trainer.max_epochs=1
poetry run python src/train.py dataset=ICSG3D_paper trainer.max_epochs=1
poetry run python src/train.py dataset=lim_l6_paper trainer.max_epochs=1

