# Demo — run the model on your own image

`demo.py` runs `predict_board` on a chessboard photo and prints the predicted 8×8 board.
It works on **any** RGB chessboard image — no dataset needed (the trained weights ship in the repo).

## 1. Setup (once)

From the project root:

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 2. Run on a single image

```bash
python demo/demo.py --input path/to/your_board.jpg
```

Prints an ASCII board and the raw `(8, 8)` tensor (values `0–12`).

## 3. Save a visualisation (input vs. predicted board)

```bash
python demo/demo.py --input path/to/your_board.jpg --save
```

Writes `your_board_predicted.png` next to the input image.

## 4. Run on a whole folder

```bash
python demo/demo.py --input path/to/folder
```

## Example (sample image included in the dataset)

```bash
python demo/demo.py --input data/game7_per_frame/images/frame_000696.jpg --save
```

## Output legend

Uppercase = white, lowercase = black, `.` = empty:

```text
P R N B Q K   white pawn / rook / knight / bishop / queen / king
p r n b q k   black pawn / rook / knight / bishop / queen / king
.             empty square
```

Class ids in the tensor: `0–5` white P/R/N/B/Q/K, `6–11` black p/r/n/b/q/k, `12` empty.

> The first run may print a few harmless `xFormers is not available` warnings — ignore them.
