---
notion-id: 28720c92-0436-80fb-bc06-ce5fcee8c88b
---
1. Hierarchical Graph Generation
    - **Leaf nodes** = atomic parts (real-world parts)
    - **Non-leaf nodes** = subassemblies
    - **Root node** = final assembled product

    - Scene to JSON
        - Use VLM with vision and text input to generate JSON representation of environment
    - Building assembly instructions
        - Use VLM to build graph in text based format (JSON based)
2. Per-step Assembly Pose Estimation
    - Encoder for pointcloud
    - 


Point clouds are given

Improvements?