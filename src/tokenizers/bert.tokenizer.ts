import path from "path";
import { Tokenizer as BaseTokenizer, Encoding, AddedToken, WordPiece } from "tokenizers";
import { FullTokenizerOptions, Tokenizer } from "./tokenizer";
import { exists } from "../utils";
import { DEFAULT_VOCAB_PATH } from "../qa-options";

export interface BertTokenizerOptions {
  clsToken: string;
  maskToken: string;
  padToken: string;
  sepToken: string;
  unkToken: string;
}

export class BertTokenizer extends Tokenizer {
  private readonly sepToken: string;

  constructor(tokenizer: BaseTokenizer, options: BertTokenizerOptions) {
    super(tokenizer);
    this.sepToken = options.sepToken;
  }

  static async fromOptions(
    options: FullTokenizerOptions<BertTokenizerOptions>
  ): Promise<BertTokenizer> {
    let vocabPath = options.vocabFile || path.join(options.filesDir, "vocab.txt");

    if (!(await exists(vocabPath))) {
      vocabPath = DEFAULT_VOCAB_PATH;
    }

    const model = await WordPiece.fromFile(vocabPath, {
      unkToken: options.unkToken || "[UNK]",
      continuingSubwordPrefix: "##",
      maxInputCharsPerWord: 100
    });

    const tokenizer = new BaseTokenizer(model);
    
    const specialTokens = [
      new AddedToken(options.clsToken || "", true),
      options.clsToken || "",
      options.maskToken || "",
      options.padToken || "",
      options.sepToken || "",
      options.unkToken || ""
    ].filter(Boolean);
    tokenizer.addTokens(specialTokens.map(token => typeof token === "string" ? token : token.getContent()));

    return new BertTokenizer(tokenizer, options as BertTokenizerOptions);
  }

  getQuestionLength(encoding: Encoding): number {
    const sepTokenIndex = encoding.getTokens().indexOf(this.sepToken);
    if (sepTokenIndex === -1) {
      return 0;
    }
    return sepTokenIndex - 1;
  }

  getContextStartIndex(encoding: Encoding): number {
    return this.getQuestionLength(encoding) + 2;
  }
}
