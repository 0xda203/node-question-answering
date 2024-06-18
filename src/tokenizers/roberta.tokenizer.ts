import path from "path";
import {
  Tokenizer as BaseTokenizer,
  AddedToken,
  robertaProcessing,
  Encoding,
  byteLevelPreTokenizer,
  BPE
} from "tokenizers";
import { exists } from "../utils";
import { FullTokenizerOptions, Tokenizer } from "./tokenizer";

export interface RobertaTokenizerOptions {
  clsToken: string;
  eosToken: string;
  maskToken: string;
  padToken: string;
  unkToken: string;
}

export class RobertaTokenizer extends Tokenizer {
  private readonly eosToken: string;
  private readonly padToken: string;

  constructor(tokenizer: BaseTokenizer, options: RobertaTokenizerOptions) {
    super(tokenizer);
    this.eosToken = options.eosToken;
    this.padToken = options.padToken;
  }

  static async fromOptions(
    options: FullTokenizerOptions<RobertaTokenizerOptions>
  ): Promise<RobertaTokenizer> {
    const vocabPath = path.join(options.filesDir, options.vocabFile || "vocab.json");
    const mergesPath = path.join(options.filesDir, options.mergesFile || "merges.txt");

    if (!(await exists(vocabPath))) {
      throw new Error("Unable to find a vocab file. Make sure to provide its path in the options");
    }

    if (!(await exists(mergesPath))) {
      throw new Error("Unable to find a merges file. Make sure to provide its path in the options");
    }

    const model = await BPE.fromFile(vocabPath, mergesPath, {
      unkToken: options.unkToken || "<unk>"
    });

    const tokenizer = new BaseTokenizer(model);

    tokenizer.setPreTokenizer(byteLevelPreTokenizer(true));

    const specialTokens = [
      new AddedToken(options.clsToken || "<s>", true),
      new AddedToken(options.eosToken || "</s>", true),
      new AddedToken(options.maskToken || "<mask>", true),
      new AddedToken(options.padToken || "<pad>", true),
      new AddedToken(options.unkToken || "<unk>", true)
    ];
    tokenizer.addSpecialTokens(specialTokens.map(token => token.getContent()));

    const eosId = tokenizer.tokenToId(options.eosToken || "</s>");
    const clsId = tokenizer.tokenToId(options.clsToken || "<s>");
    if (eosId === null || clsId === null) {
      throw new Error("CLS or EOS tokens are not part of the vocabulary.");
    }
    tokenizer.setPostProcessor(robertaProcessing([options.eosToken || "</s>", eosId], [options.clsToken || "<s>", clsId]));

    return new RobertaTokenizer(tokenizer, {
      clsToken: options.clsToken || "<s>",
      eosToken: options.eosToken || "</s>",
      maskToken: options.maskToken || "<mask>",
      padToken: options.padToken || "<pad>",
      unkToken: options.unkToken || "<unk>"
    });
  }

  getQuestionLength(encoding: Encoding): number {
    const eosTokenIndex = encoding.getTokens().indexOf(this.eosToken);
    return eosTokenIndex !== -1 ? eosTokenIndex - 1 : 0;
  }

  getContextStartIndex(encoding: Encoding): number {
    return this.getQuestionLength(encoding) + 2;
  }

  setPadding(maxLength: number): void {
    const padId = this.tokenizer.tokenToId(this.padToken);
    if (padId === null) {
      throw new Error("Pad token must be part of the vocabulary.");
    }
    this.tokenizer.setPadding({ maxLength, padId, padToken: this.padToken });
  }
}
