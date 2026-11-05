# Makemore

Run the terminal chat interface using the saved checkpoint:

```sh
uv run python nano-gpt/chat.py
```

Generation controls:

```sh
uv run python nano-gpt/chat.py --tokens 500 --temperature 0.8 --top-k 20
```

Use `/reset` to clear conversation history and `/quit` to exit. Replies stream
one character at a time. The last 256 characters provide context for each reply.
The checkpoint is a Shakespeare character model, so replies are text
continuations rather than instruction-following assistant answers. Characters
outside its training vocabulary are rejected with an explanation.

The checkpoint defaults to `nano-gpt/model.pth` relative to the script. Use
`--checkpoint PATH` or `--device cpu` to override the model file or device.
