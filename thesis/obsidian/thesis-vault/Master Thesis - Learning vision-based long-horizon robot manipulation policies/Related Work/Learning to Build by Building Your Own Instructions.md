---
notion-id: 28720c92-0436-80c4-bdb2-c8a4f565a024
---
→ Only step 3 is relevant for our purpose but this architecture enables more training data as disassembly is easier than assembly

1. We have assembled models
2. We disassemble to create instruction images
    1. Model decides how to disassemble
    2. Model uses cursor to point to parts to remove
    3. Model doesnt have disassembly instruction
3. We assemble again and train model to do this

→ Our images stack is the assembly instruction

Online learning


Build a lego set in a single interactive session

LTRON → previously proposed make and break problem for LEGO assembly

Agent: InstructioNet

InstructioNet