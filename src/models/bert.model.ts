import { Encoding } from "tokenizers";

import { Logits, Model, ModelInput, ModelType } from "./model";

export class BertModel extends Model {
  public static readonly inputs: ReadonlyArray<ModelInput> = [
    ModelInput.AttentionMask,
    ModelInput.Ids,
    ModelInput.TokenTypeIds
  ];

  public readonly type = ModelType.Bert;

  runInference(encodings: Encoding[]): Promise<[Logits, Logits]> {
    return this.runtime.runInference(
      encodings.map(e => e.getIds()),
      encodings.map(e => e.getAttentionMask()),
      encodings.map(e => e.getTypeIds())
    );
  }
}
