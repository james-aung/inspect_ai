from inspect_ai import Task, task, eval
from inspect_ai.dataset import Sample
from inspect_ai.scorer import includes
from inspect_ai.solver import basic_agent
from inspect_ai.tool import read_file_chunk, search_file

@task
def james():
    return Task(
        dataset=[
            Sample(
                input="Please read the paper.md file and summarize what StateMask, Theorem 3.6, and Algorith 2 are and provide references to the relevant sections. Read multiple pages to get a comprehensive understanding. After that then just read the whole paper and give me a summary of it.",
                files={
                    "paper.md": "/Users/james.aung/code/openai/alcatraz-dev/project/100-papers/papers/data/papers/rice/paper.md"
                }
            )
        ],
        solver=basic_agent(
            tools=[read_file_chunk(), search_file()],
            max_attempts=1,
        ),
        time_limit=60 * 2,
        sandbox="local",
    )
    
eval(james(), model="openai/gpt-4o", trace=True)
