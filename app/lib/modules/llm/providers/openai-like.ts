import { BaseProvider, getOpenAILikeModel } from '~/lib/modules/llm/base-provider';
import type { ModelInfo } from '~/lib/modules/llm/types';
import type { IProviderSetting } from '~/types/model';
import type { LanguageModelV1 } from 'ai';
import { logger } from '~/utils/logger';

interface OpenAIModelsResponse {
  data: Array<{ id: string }>;
}

export default class OpenAILikeProvider extends BaseProvider {
  name = 'OpenAILike';
  getApiKeyLink = undefined;

  config = {
    baseUrlKey: 'OPENAI_LIKE_API_BASE_URL',
    apiTokenKey: 'OPENAI_LIKE_API_KEY',
    modelsKey: 'OPENAI_LIKE_API_MODELS',
  };

  staticModels: ModelInfo[] = [
    { name: 'glm-5.2', label: 'GLM-5.2', provider: 'OpenAILike', maxTokenAllowed: 128000 },
    { name: 'glm-5', label: 'GLM-5', provider: 'OpenAILike', maxTokenAllowed: 128000 },
    { name: 'kat-coder-pro-v2.5', label: 'KAT Coder Pro v2.5', provider: 'OpenAILike', maxTokenAllowed: 128000 },
    { name: 'DeepSeek-V4-Flash', label: 'DeepSeek V4 Flash', provider: 'OpenAILike', maxTokenAllowed: 128000 },
    { name: 'DeepSeek-V4-Pro', label: 'DeepSeek V4 Pro', provider: 'OpenAILike', maxTokenAllowed: 128000 },
    { name: 'Kimi-K2.6', label: 'Kimi K2.6', provider: 'OpenAILike', maxTokenAllowed: 128000 },
    { name: 'MiniMax-M3', label: 'MiniMax M3', provider: 'OpenAILike', maxTokenAllowed: 128000 },
    { name: 'Qwen3.6-27B', label: 'Qwen3.6 27B', provider: 'OpenAILike', maxTokenAllowed: 128000 },
    { name: 'sensenova-6.7-flash-lite', label: 'SenseNova 6.7 Flash Lite', provider: 'OpenAILike', maxTokenAllowed: 128000 },
    { name: 'sensenova-u1-fast', label: 'SenseNova U1 Fast', provider: 'OpenAILike', maxTokenAllowed: 128000 },
    { name: 'auto', label: 'Auto (Model Routing)', provider: 'OpenAILike', maxTokenAllowed: 128000 },
    { name: 'glm-4.7', label: 'GLM-4.7', provider: 'OpenAILike', maxTokenAllowed: 128000 },
    { name: 'kat-coder-pro-v2', label: 'KAT Coder Pro v2', provider: 'OpenAILike', maxTokenAllowed: 128000 },
    { name: 'deepseek-v3.2', label: 'DeepSeek V3.2', provider: 'OpenAILike', maxTokenAllowed: 128000 },
  ];

  async getDynamicModels(
    apiKeys?: Record<string, string>,
    settings?: IProviderSetting,
    serverEnv: Record<string, string> = {},
  ): Promise<ModelInfo[]> {
    const { baseUrl, apiKey } = this.getProviderBaseUrlAndKey({
      apiKeys,
      providerSettings: settings,
      serverEnv,
      defaultBaseUrlKey: 'OPENAI_LIKE_API_BASE_URL',
      defaultApiTokenKey: 'OPENAI_LIKE_API_KEY',
    });

    if (!baseUrl || !apiKey) {
      return this.staticModels;
    }

    try {
      const response = await fetch(`${baseUrl}/models`, {
        headers: {
          Authorization: `Bearer ${apiKey}`,
        },
        signal: this.createTimeoutSignal(),
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      const res = (await response.json()) as OpenAIModelsResponse;

      const dynamicModels = (res.data || []).map((model) => ({
        name: model.id,
        label: model.id,
        provider: this.name,
        maxTokenAllowed: 128000,
      }));

      const dynamicIds = new Set(dynamicModels.map((m) => m.name));
      const filteredStatic = this.staticModels.filter((m) => !dynamicIds.has(m.name));

      return [...filteredStatic, ...dynamicModels];
    } catch (error) {
      logger.info(`${this.name}: Could not fetch /models endpoint, checking fallback env`, error);

      // Fallback to OPENAI_LIKE_API_MODELS if available
      // eslint-disable-next-line dot-notation
      const modelsEnv = serverEnv['OPENAI_LIKE_API_MODELS'] || settings?.OPENAI_LIKE_API_MODELS;

      if (modelsEnv) {
        logger.info(`${this.name}: Using OPENAI_LIKE_API_MODELS fallback`);

        return this._parseModelsFromEnv(modelsEnv);
      }

      return this.staticModels;
    }
  }

  /**
   * Parse OPENAI_LIKE_API_MODELS environment variable
   * Format: path/to/model1:limit;path/to/model2:limit;path/to/model3:limit
   */
  private _parseModelsFromEnv(modelsEnv: string): ModelInfo[] {
    if (!modelsEnv) {
      return [];
    }

    try {
      const models: ModelInfo[] = [];
      const modelEntries = modelsEnv.split(';');

      for (const entry of modelEntries) {
        const trimmedEntry = entry.trim();

        if (!trimmedEntry) {
          continue;
        }

        const [modelPath, limitStr] = trimmedEntry.split(':');

        if (!modelPath) {
          continue;
        }

        const limit = limitStr ? parseInt(limitStr.trim(), 10) : 8000;
        const modelName = modelPath.trim();

        // Generate a readable label from the model path
        const label = this._generateModelLabel(modelName);

        models.push({
          name: modelName,
          label,
          provider: this.name,
          maxTokenAllowed: limit,
        });
      }

      logger.info(`${this.name}: Parsed ${models.length} models from env`);

      return models;
    } catch (error) {
      logger.error(`${this.name}: Error parsing OPENAI_LIKE_API_MODELS:`, error);
      return [];
    }
  }

  /**
   * Generate a readable label from model path
   */
  private _generateModelLabel(modelPath: string): string {
    // Extract the last part of the path and clean it up
    const parts = modelPath.split('/');
    const lastPart = parts[parts.length - 1];

    // Remove common prefixes and clean up the name
    let label = lastPart
      .replace(/^accounts\//, '')
      .replace(/^fireworks\/models\//, '')
      .replace(/^models\//, '')
      // Capitalize first letter of each word
      .replace(/\b\w/g, (l) => l.toUpperCase())
      // Replace spaces with hyphens for a cleaner look
      .replace(/\s+/g, '-');

    // Add provider suffix if not already present
    if (!label.includes('Fireworks') && !label.includes('OpenAI')) {
      label += ' (OpenAI Compatible)';
    }

    return label;
  }

  getModelInstance(options: {
    model: string;
    serverEnv: Env;
    apiKeys?: Record<string, string>;
    providerSettings?: Record<string, IProviderSetting>;
  }): LanguageModelV1 {
    const { model, serverEnv, apiKeys, providerSettings } = options;
    const envRecord = this.convertEnvToRecord(serverEnv);

    const { baseUrl, apiKey } = this.getProviderBaseUrlAndKey({
      apiKeys,
      providerSettings: providerSettings?.[this.name],
      serverEnv: envRecord,
      defaultBaseUrlKey: 'OPENAI_LIKE_API_BASE_URL',
      defaultApiTokenKey: 'OPENAI_LIKE_API_KEY',
    });

    if (!baseUrl || !apiKey) {
      throw new Error(`Missing configuration for ${this.name} provider`);
    }

    return getOpenAILikeModel(baseUrl, apiKey, model);
  }
}
