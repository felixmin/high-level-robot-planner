---
notion-id: 29a20c92-0436-808e-861e-ebe1eca5dd1e
---
We want to test how good ChatGPT can do the planning out of the box. We give ChatGPT a current camera observation and a building instruction and ask for the next step.

Results

Takeaway

> [!note]+ Prompt
> <INSTRUCTION> We are building a LEGO model. Give me the single next step in the building process. The steps should be very fine-grained instructions. Also check if we are on the right track and undo or make adjustments otherwise. <INSTRUCTION> 
> 
> <OBSERVATION> image attached </OBSERVATION> 
> <GOAL> image attached </GOAL> 
> 
> <EXAMPLES>
>  <EXAMPLE> Take the yellow square block on the left side of the block pile. </EXAMPLE> 
> <EXAMPLE> Place the red block which you currently have in your hand with two overlapping pins with the yellow block square and two overlapping with the green 6x2 block on the long side. </EXAMPLE> 
> </EXAMPLES>
> 
> !![[image 7.png]]

> [!note]+ Step 1
> !![[image 8.png]]
> 
> Take the long flat yellow **2×8 plate** from the pile and set it horizontally on the table with studs facing up. If you grabbed a tall 2×8 brick instead, put it back—we need the flat plate for the base.

> [!note]+ Step 2
> !![[image 9.png]]
> 
> Pick the yellow 2×2 brick with the smiling face and snap it onto the **rightmost two studs** of the long yellow plate, aligned to the front edge. You're on track.

> [!note]+ Step 3
> !![[image 10.png]]
> 
> Find the **yellow 2×4 rounded brick** (oval ends, flat top). **Place it lengthwise on the base plate directly behind the head, centered across the two rows, touching the head.** Push down to lock all 4 studs.

> [!note]+ Step 4
> !![[image 11.png]]
> 
> Tries:
> 
> - Pick up the flat light-tan 2×8 plate from the pile and place it horizontally in front of you, studs up.
> - Undo the last bit: detach the tall yellow **2×2 brick** from the right end of your base so the base is **only two flat 2×4 yellow plates connected lengthwise**.
> - Flip the long light-yellow 2×8 plate on your notebook so the studs face up, and lay it horizontally in front of you as the base. (You’re on the right track—this is the correct base piece.)
> 
> !![[image 12.png]]
> 
> Tries: 
> 
> - Detach the yellow 2×2 face brick from the side, turn its face forward, and place it on the **rightmost two studs** of the long base plate—directly in front of the stacked 2×4 bricks.
> - Remove the plain yellow 2×4 rectangular brick that’s in front of the face block on the base.
> - Take apart the small yellow stack sitting on the notebook—those pieces aren’t used yet for this step.
> - Find the yellow 2×8 plate and place it flat in front of you, studs up, with the long side left-to-right.
> - Remove the yellow 2×4 rectangular brick on top and snap on the yellow curved 2×4 “roof” piece with four studs, centered over the rounded shell.

Experiments

- Desired part is not in the pile
- LAPA helps us in not needing language as output 
