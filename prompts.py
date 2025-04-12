# QUERY_PREFIX = """Regarding a given video, use tools to answer the following question as best you can.

# Question: """

QUERY_PREFIX = """Question: """

ASSISTANT_ROLE = """You are an AI assistant for video analysis. Regarding a given video, you will receive information from sampled frames, use tools to extract additional information, and answer the following question as accurately as possible.
"""

TOOLS_RULE = """Please strictly follow the rules below while using the tools:
Rule 1: Do not use the frame-extraction-tool as the first tool. 
Rule 2: If, after using other tools, you still do not have enough information to provide a clear answer, you must use the frame-extraction-tool to extract more frames from the video.
Rule 3: After using the frame-extraction-tool, please continue with other tools to analyze the extracted frames."
Rule 4: The same tool should not be invoked consecutively.
"""


SELECT_FRAMES_PROMPT = """Given a video that has {num_frames} frames, the frames are decoded at {fps} fps. 

Given the following information of sampled frames in the video:
```
{visible_frames_info}
```

To answer the following question: 
``` 
{question}
```

However, the information in the initial sampled frames is not suffient. Our goal is to identify additional frames that contain crucial information necessary for answering the question. These frames should not only address the query directly but should also complement the insights gleaned from the descriptions of the initial frames.

To achieve this, we will:

1. List the uninformed video segments between sampled frames in the format 'segment_id': 'start_frame_index'-'end_frame_index': 
```
{candidate_segment}
```

2. Determine which segments are likely to contain frames that are most relevant to the question. These frames should capture key visual elements, such as objects, humans, interactions, actions, and scenes, that are supportive to answer the question.

Return the selected video segments in the specified JSON format.
"""

QUERY_PREFIX_DES = """Regarding a given video, based on the following frame caption to answer the following question as best you can.

Frame Caption: 
{frame_caption}

Question:
{question}
"""