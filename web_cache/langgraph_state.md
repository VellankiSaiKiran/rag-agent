# LangGraph state

In LangGraph, an agent is built as a graph of steps, called nodes, connected by edges. The state is the shared record that flows through this graph. It holds everything the agent knows at a given moment, such as the user's question, intermediate results, and the final answer.

The state is usually described with a typed dictionary that lists each field and its type. Every node receives the current state as input. Instead of returning a whole new state, a node returns only the fields it wants to change, and LangGraph merges that update into the existing state.

How each field is merged depends on its reducer. By default, a new value simply replaces the old one. A field can instead be given a reducer such as list addition, so each update appends to the list rather than overwriting it. This is how an agent can collect results across many steps or many loops.

Edges can read the state to decide where to go next. A routing function looks at the current values, for example whether enough evidence has been gathered, and returns the name of the next node. Because routing depends only on the state, it is easy to test by handing it different states and checking the result.
