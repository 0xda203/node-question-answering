import {
  Encoding,
  TruncationOptions,
  Tokenizer as BaseTokenizer
} from "tokenizers";

import { ModelType } from "../models";

export interface TokenizerBaseOptions {
  /**
   * @default true
   */
  lowercase?: boolean;
  /**
   * Name of the merges file (if applicable to the tokenizer)
   * @default "merges.txt"
   */
  mergesFile?: string;
  /**
   * Directory under which the files needed by the tokenizer are located.
   * Must be an absolute path.
   */
  filesDir: string;
  modelType: ModelType;
  /**
   * Name of the vocab file (if applicable to the tokenizer)
   * @default "vocab.txt" | "vocab.json"
   */
  vocabFile?: string;
}

export type FullTokenizerOptions<TokSpecificOptions> = TokenizerBaseOptions & 
  Partial<TokSpecificOptions>;

export abstract class Tokenizer<T extends BaseTokenizer = BaseTokenizer> {
  protected tokenizer: T;

  constructor(tokenizer: T) {
    this.tokenizer = tokenizer;
  }

  abstract getQuestionLength(encoding: Encoding): number;

  abstract getContextStartIndex(encoding: Encoding): number;

  /**
   * Get the last index of the context of an encoding
   * @param encoding Encoding for which to return last context index
   * @virtual
   */
  getContextEndIndex(encoding: Encoding): number {
    const nbAddedTokens = encoding.getSpecialTokensMask().reduce((acc, val) => acc + val, 0);
    const actualLength = encoding.getLength() - nbAddedTokens;
    const contextLength = actualLength - this.getQuestionLength(encoding);

    return this.getContextStartIndex(encoding) + contextLength - 1;
  }

  encode(sequence: string, pair?: string, addSpecialTokens = true): Promise<Encoding> {
    return this.tokenizer.encode(sequence, pair, { addSpecialTokens });
  }

  /**
   * Enable/change padding with specified options
   * @param maxLength Padding length
   * @virtual
   */
  setPadding(maxLength: number) {
    this.tokenizer.setPadding({ maxLength });
  }

  setTruncation(
    maxLength: number,
    options?: TruncationOptions
  ) {
    this.tokenizer.setTruncation(maxLength, options);
  }
}
