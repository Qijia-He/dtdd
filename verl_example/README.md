# verl OPD Example

This directory provides an example script for running On-Policy Distillation (OPD) with [verl](https://github.com/volcengine/verl). Please install **verl v0.8.0** or later before launching the script.

## Configuration Notes

Select the OPD variant by setting the following environment variables before running `opd.sh`.

For sampled-token OPD:

```bash
export DISTILLATION_LOSS_MODE=k1
export USE_POLICY_GRADIENT=True
```

For top-k KL OPD:

```bash
export DISTILLATION_LOSS_MODE=forward_kl_topk
export USE_POLICY_GRADIENT=False
```

Then start the example with:

```bash
bash verl_example/opd.sh
```
