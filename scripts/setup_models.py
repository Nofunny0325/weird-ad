"""One-time download of public model weights. No ad data is read or submitted."""
import argparse
from hashlib import sha256
from pathlib import Path
from urllib.request import Request, urlopen
import tempfile

ROOT = Path(__file__).resolve().parents[1]
E5_REVISION = '614241f622f53c4eeff9890bdc4f31cfecc418b3'
E5_FILES = {
    'model.onnx': ('model_qint8_avx512_vnni.onnx','dd476dd0c2514e9b9be83aeb3853fac0763e0bdf4a71645407587d77c48a2d88'),
    'tokenizer.json': ('tokenizer.json','0b44a9d7b51c3c62626640cda0e2c2f70fdacdc25bbbd68038369d14ebdf4c39'),
}
WHISPER_REVISIONS = {
    'small': '536b0662742c02347bc0e980a01041f333bce120',
    'base': 'ebe41f70d5b6dfa9166e2c581c45c9c0cfc57b66',
}


def checksum(path):
    digest=sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):
            digest.update(chunk)
    return digest.hexdigest()


def setup_e5(directory):
    directory.mkdir(parents=True,exist_ok=True)
    for filename,(remote,digest) in E5_FILES.items():
        path=directory/filename
        if path.is_file() and checksum(path)==digest:
            print(f'Already verified: {path}')
            continue
        url=f'https://huggingface.co/intfloat/multilingual-e5-small/resolve/{E5_REVISION}/onnx/{remote}?download=true'
        with tempfile.TemporaryDirectory(dir=directory) as tmp:
            candidate=Path(tmp)/filename
            with urlopen(Request(url,headers={'User-Agent':'weird-ad-model-setup/1.0'}),timeout=120) as source, candidate.open('wb') as dest:
                for chunk in iter(lambda:source.read(1024*1024),b''):
                    dest.write(chunk)
            if checksum(candidate)!=digest:
                raise RuntimeError(f'Model SHA-256 mismatch: {filename}')
            candidate.replace(path)
        print(f'Installed and verified: {path}')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--e5-dir',type=Path,default=ROOT/'storage'/'models'/'e5')
    parser.add_argument('--with-stt',action='store_true',help='Also install the optional Whisper model')
    parser.add_argument('--stt-model',choices=WHISPER_REVISIONS,default='small',
                        help='Whisper model to install when --with-stt is enabled')
    args=parser.parse_args()
    setup_e5(args.e5_dir)
    if args.with_stt:
        from huggingface_hub import snapshot_download
        model = args.stt_model
        snapshot_download(f'Systran/faster-whisper-{model}',revision=WHISPER_REVISIONS[model],
            local_dir=ROOT/'storage'/'models'/f'whisper-{model}',
            allow_patterns=['config.json','model.bin','tokenizer.json','vocabulary.*'],token=False)


if __name__=='__main__':
    main()
