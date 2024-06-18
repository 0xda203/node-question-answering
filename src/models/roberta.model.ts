import { Encoding } from "tokenizers";

import { Logits, Model, ModelInput, ModelType } from "./model";

export class RobertaModel extends Model {
  public static readonly inputs: ReadonlyArray<ModelInput> = [
    ModelInput.AttentionMask,
    ModelInput.Ids
  ];

  public readonly type = ModelType.Roberta;

  runInference(encodings: Encoding[]): Promise<[Logits, Logits]> {
    return this.runtime.runInference(
      encodings.map(e => e.getIds()),
      encodings.map(e => e.getAttentionMask()),
    );
  }
}
