{
      "name": "buf10",
      "node": {
        "target": "fb::fn_with_mix_outputs",
        "inputs": [
          {
            "name": "t",
            "arg": {
              "asTensor": {
                "name": "buf8"
              }
            }
          },
          {
            "name": "tensors",
            "arg": {
              "asTensors": [
                {
                  "name": "buf9"
                },
                {
                  "name": "buf6"
                }
              ]
            }
          }
        ],
        "outputs": [
          {
            "asTensor": {
              "name": "buf11"
            }
          },
          {
            "asTensors": [
              {
                "name": "buf12"
              },
              {
                "name": "buf13"
              }
            ]
          }
        ],
        "metadata": {}
      }
    }