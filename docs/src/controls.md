## Controls

Controls are repeatedly applied in sequence until a control triggers a
stop. Each control type has a detailed doc-string. Below is a short
summary, with some advanced options omitted.

| control                                       | description                                                                             | enabled if these are overloaded     | can trigger a stop | notation in Prechelt |
|-----------------------------------------------|-----------------------------------------------------------------------------------------|-------------------------------------|--------------------|----------------------|
| [`Step(n=1)`](@ref)                           | Train model for `n` iterations                                                          | `train!`                            | no                 |                      |
| [`Info(f=identity)`](@ref)                    | Log to `Info` the value of `f(model)`                                                   | `train!`                            | no                 |                      |
| [`Warn(predicate, f="")`](@ref)               | Log to `Warn` the value of `f` or `f(model)` if `predicate(model)` holds                | `train!`                            | no                 |                      |
| [`Error(predicate, f="")`](@ref)              | Log to `Error` the value of `f` or `f(model)` if `predicate(model)` holds and then stop | `train!`                            | yes                |                      |
| [`Callback(f=identity)`](@ref)                | Call `f(model)`                                                                         | `train!`                            | yes                |                      |
| [`TimeLimit(t=0.5)`](@ref)                    | Stop after `t` hours                                                                    | `train!`                            | yes                |                      |
| [`NumberLimit(n=100)`](@ref)                  | Stop after `n` applications of the control                                              | `train!`                            | yes                |                      |
| [`NumberSinceBest(n=6)`](@ref)                | Stop when best loss occurred `n` control applications ago                               | `train!`                            | yes                |                      |
| [`WithNumberDo(f=identity)`](@ref)            | Call `f(n + 1)` where `n` is the number of complete control cycles so far               | `train!`                            | yes                |                      |
| [`WithLossDo(f=x->@info("loss: $x"))`](@ref)  | Call `f(loss)` where `loss` is the current loss                                         | `train!`, `loss`                    | yes                |                      |
| [`WithTrainingLossesDo(f=v->@info(v))`](@ref) | Call `f(v)` where `v` is the current batch of training losses                           | `train!`, `training_losses`         | yes                |                      |
| [`InvalidValue()`](@ref)                      | Stop when `NaN`, `Inf` or `-Inf` loss/training loss encountered                         | `train!`                            | yes                |                      |
| [`Threshold(value=0.0)`](@ref)                | Stop when `loss < value`                                                                | `train!`, `loss`                    | yes                |                      |
| [`GL(alpha=2.0)`](@ref)                       | Stop after "Generalization Loss" exceeds `alpha`                                        | `train!`, `loss`                    | yes                | ``GL_α``             |
| [`Patience(n=5)`](@ref)                       | Stop after `n` consecutive loss increases                                               | `train!`, `loss`                    | yes                | ``UP_s``             |
| [`PQ(alpha=0.75, k=5)`](@ref)                 | Stop after "Progress-modified GL" exceeds `alpha`                                       | `train!`, `loss`, `training_losses` | yes                | ``PQ_α``             |
| [`Warmup(c; n=1)`](@ref)                      | Wait for `n` loss updates before checking criteria `c`                                  | `train!`                            | no                 |                      |
| [`Data(data)`](@ref)                          | Call `ingest!(model, datum)` on the next `item` in the iterable `data`.                 | `train!`, `ingest!`                 | yes                |                      |
|                                               |                                                                                         |                                     |                    |                      |

> Table 1. Atomic controls

**Stopping option.** All the following controls trigger a stop if the
provided function `f` returns `true` and `stop_if_true=true` is
specified in the constructor: `Callback`, `WithNumberDo`, `WithLossDo`,
`WithTrainingLossesDo`.

There are also three control wrappers to modify a control's behavior:

| wrapper                                                    | description                                                                                                                                  |
|------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------|
| [`IterationControl.skip(control; predicate=1)`](@ref)      | Apply `control` every `predicate` applications of the control wrapper (can also be a function; see doc-string)                               |
| [`IterationControl.louder(control; by=1)`](@ref)           | Increase the verbosity level of `control` by the specified value (negative values lower verbosity)                                           |
| [`IterationControl.with_state_do`](@ref)`(control; f=...)` | Apply control *and* call `f(x)` where `x` is the internal (public) state of control; useful for debugging. Default `f` logs state to `Info`. |
| [`IterationControl.composite(controls...)`](@ref)          | Apply each `control` in `controls` in sequence; mostly for internal use                                                                      |

> Table 2. Wrapped controls

```@docs
IterationControl.Step
IterationControl.Info
IterationControl.Warn
IterationControl.Error
IterationControl.Callback
EarlyStopping.TimeLimit
EarlyStopping.NumberLimit
EarlyStopping.NumberSinceBest
IterationControl.WithNumberDo
IterationControl.WithLossDo
IterationControl.WithTrainingLossesDo
EarlyStopping.InvalidValue
EarlyStopping.Threshold
EarlyStopping.GL
EarlyStopping.Patience
EarlyStopping.PQ
EarlyStopping.Warmup
IterationControl.Data
IterationControl.skip
IterationControl.louder
IterationControl.with_state_do
IterationControl.composite
```
