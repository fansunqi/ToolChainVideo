from prompts import SELECT_FRAMES_PROMPT

select_frames_prompt = SELECT_FRAMES_PROMPT.format(
    num_frames = 30,
    fps = 1,
    caption = "",
    question = "",
    segment_des = "",
    formatted_description = "",
    segment_id_max = ""
)

print(select_frames_prompt)

