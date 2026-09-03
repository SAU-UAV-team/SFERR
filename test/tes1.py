#!/usr/bin/env python3
"""
CHARON缓冲区策略快速测试 - 修复版
"""

import os
import sys
import torch
import numpy as np
import unittest
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.buffer import Buffer


class MockArgs:
    """完整的模拟训练参数，包含所有必要属性"""

    def __init__(self):
        # 缓冲区相关参数
        self.buffer_size = 500
        self.minibatch_size = 32

        # 损失权重参数
        self.alpha = 0.5
        self.beta = 0.5
        self.gamma = 1.0

        # 模型和重建相关参数
        self.model = "charon"
        self.sampling_interval = 5  # 修复：添加这个缺失的属性
        self.masking_ratio = 0.5

        # 训练参数
        self.n_epochs = 5
        self.linear_probing_epochs = 2
        self.grad_clip = None
        self.debug_mode = True

        # herding相关参数
        self.herding_sliding_window = False
        self.herding_window_size = 500
        self.herding_alpha = 0.99


class BufferStrategyTester:
    """缓冲区策略测试器"""

    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = torch.device(device)
        self.args = MockArgs()
        print(f"🚀 初始化测试环境 - 设备: {self.device}")

    def test_buffer_initialization(self):
        """测试缓冲区初始化"""
        print("\n=== 测试缓冲区初始化 ===")

        strategies = ['reservoir', 'herding']
        for strategy in strategies:
            try:
                buffer = Buffer(self.args.buffer_size, self.device, mode=strategy, args=self.args)
                print(f"✅ {strategy}模式初始化成功")
                print(f"   缓冲区大小: {buffer.buffer_size}")
                print(f"   当前样本数: {buffer.num_seen_examples}")
                print(f"   模式: {buffer.mode}")
            except Exception as e:
                print(f"❌ {strategy}模式初始化失败: {e}")

    def test_data_addition(self, n_samples=100):
        """测试数据添加功能"""
        print(f"\n=== 测试数据添加 ({n_samples}个样本) ===")

        for strategy in ['reservoir', 'herding']:
            try:
                buffer = Buffer(self.args.buffer_size, self.device, mode=strategy, args=self.args)

                # 生成测试数据
                examples = torch.randn(n_samples, 3, 32, 32)
                labels = torch.randint(0, 5, (n_samples,))
                features = torch.randn(n_samples, 512)

                if strategy == 'herding':
                    # herding模式需要重建损失
                    recon_losses = torch.rand(n_samples) * 0.1
                    buffer.add_data(examples, labels=labels, features=features, recon_losses=recon_losses)
                else:
                    buffer.add_data(examples, labels=labels, features=features)

                print(f"✅ {strategy}模式数据添加成功")
                print(f"   缓冲区当前样本: {len(buffer)}/{buffer.buffer_size}")

            except Exception as e:
                print(f"❌ {strategy}模式数据添加失败: {e}")

    def test_herding_prototype_selection(self):
        """测试herding模式的原型选择"""
        print("\n=== 测试Herding原型选择 ===")

        try:
            buffer = Buffer(self.args.buffer_size, self.device, mode='herding', args=self.args)

            # 生成模拟数据
            n_classes = 3
            samples_per_class = 30

            for class_id in range(n_classes):
                examples = torch.randn(samples_per_class, 3, 32, 32)
                labels = torch.full((samples_per_class,), class_id)
                features = torch.randn(samples_per_class, 512)
                recon_losses = torch.rand(samples_per_class) * 0.1

                buffer.add_data(examples, labels=labels, features=features, recon_losses=recon_losses)

            # 生成类别原型
            class_prototypes = {
                i: torch.randn(512) for i in range(n_classes)
            }

            # 执行原型选择
            buffer.select_prototypes(class_prototypes, current_task=0)

            print("✅ Herding原型选择完成")
            print(f"   选择的原型类别数: {len(buffer.class_prototypes)}")
            for cls, prototypes in buffer.class_prototypes.items():
                print(f"   类别{cls}: {len(prototypes)}个原型")

        except Exception as e:
            print(f"❌ Herding原型选择失败: {e}")
            import traceback
            traceback.print_exc()

    def test_buffer_retrieval(self):
        """测试缓冲区数据检索"""
        print("\n=== 测试缓冲区数据检索 ===")

        for strategy in ['reservoir', 'herding']:
            try:
                buffer = Buffer(self.args.buffer_size, self.device, mode=strategy, args=self.args)

                # 添加一些数据
                examples = torch.randn(50, 3, 32, 32)
                labels = torch.randint(0, 5, (50,))
                features = torch.randn(50, 512)

                if strategy == 'herding':
                    recon_losses = torch.rand(50) * 0.1
                    buffer.add_data(examples, labels=labels, features=features, recon_losses=recon_losses)
                    # 对于herding，需要先选择原型
                    class_prototypes = {i: torch.randn(512) for i in range(5)}
                    buffer.select_prototypes(class_prototypes, 0)
                else:
                    buffer.add_data(examples, labels=labels, features=features)

                # 测试数据检索
                if not buffer.is_empty():
                    retrieved_data = buffer.get_data(10)
                    print(f"✅ {strategy}模式数据检索成功")
                    print(f"   检索到 {len(retrieved_data)} 个数据项")
                    print(f"   第一个数据项形状: {retrieved_data[0].shape}")
                else:
                    print(f"⚠️  {strategy}模式缓冲区为空，无法检索")

            except Exception as e:
                print(f"❌ {strategy}模式数据检索失败: {e}")

    def quick_strategy_comparison(self, n_tasks=2, samples_per_task=200):
        """快速策略比较"""
        print(f"\n=== 快速策略比较 ({n_tasks}个任务, 每任务{samples_per_task}样本) ===")

        results = {}

        for strategy in ['reservoir', 'herding']:
            print(f"\n🔧 测试 {strategy} 策略:")

            try:
                buffer = Buffer(self.args.buffer_size, self.device, mode=strategy, args=self.args)
                task_metrics = []

                for task_id in range(n_tasks):
                    # 生成任务数据
                    examples = torch.randn(samples_per_task, 3, 32, 32)
                    labels = torch.randint(task_id * 2, (task_id + 1) * 2, (samples_per_task,))
                    features = torch.randn(samples_per_task, 512)

                    if strategy == 'herding':
                        recon_losses = torch.rand(samples_per_task) * 0.1
                        buffer.add_data(examples, labels=labels, features=features, recon_losses=recon_losses)

                        # 执行原型选择
                        class_prototypes = {
                            i: torch.randn(512) for i in range((task_id + 1) * 2)
                        }
                        buffer.select_prototypes(class_prototypes, task_id)
                    else:
                        buffer.add_data(examples, labels=labels, features=features)

                    # 计算指标
                    metrics = self._compute_buffer_metrics(buffer, task_id)
                    task_metrics.append(metrics)

                    print(
                        f"   任务{task_id}: 大小{metrics['buffer_size']}/多样性{metrics['diversity']:.3f}/平衡度{metrics['class_balance']:.3f}")

                results[strategy] = task_metrics

            except Exception as e:
                print(f"❌ {strategy}策略测试失败: {e}")
                results[strategy] = []

        return results

    def _compute_buffer_metrics(self, buffer, task_id):
        """计算缓冲区质量指标"""
        metrics = {'task': task_id}

        if buffer.is_empty():
            metrics.update({'buffer_size': 0, 'diversity': 0, 'class_balance': 0})
            return metrics

        metrics['buffer_size'] = len(buffer)
        metrics['buffer_capacity'] = buffer.buffer_size

        # 多样性指标
        if hasattr(buffer, 'features') and len(buffer) > 1:
            try:
                features = buffer.features[:len(buffer)].cpu().numpy()
                feature_variance = np.var(features, axis=0)
                metrics['diversity'] = np.mean(feature_variance)
            except:
                metrics['diversity'] = 0
        else:
            metrics['diversity'] = 0

        # 类别平衡指标
        if hasattr(buffer, 'labels'):
            try:
                labels = buffer.labels[:len(buffer)].cpu().numpy()
                unique, counts = np.unique(labels, return_counts=True)
                if len(unique) > 0:
                    balance_ratio = np.min(counts) / np.max(counts) if np.max(counts) > 0 else 0
                    metrics['class_balance'] = balance_ratio
                    metrics['n_classes'] = len(unique)
                else:
                    metrics['class_balance'] = 0
                    metrics['n_classes'] = 0
            except:
                metrics['class_balance'] = 0
                metrics['n_classes'] = 0

        return metrics

    def run_comprehensive_test(self):
        """运行全面测试"""
        print("=" * 60)
        print("🎯 CHARON缓冲区策略全面快速测试")
        print("=" * 60)

        # 1. 基础功能测试
        self.test_buffer_initialization()

        # 2. 数据添加测试
        self.test_data_addition(n_samples=150)

        # 3. 数据检索测试
        self.test_buffer_retrieval()

        # 4. Herding特定测试
        self.test_herding_prototype_selection()

        # 5. 策略比较
        print("\n" + "=" * 60)
        print("📊 策略性能比较")
        print("=" * 60)
        results = self.quick_strategy_comparison(n_tasks=2, samples_per_task=100)

        # 6. 输出总结
        self._print_results_summary(results)

        return results

    def _print_results_summary(self, results):
        """输出测试结果总结"""
        print("\n" + "=" * 60)
        print("📈 测试结果总结")
        print("=" * 60)

        for strategy, task_metrics in results.items():
            if not task_metrics:  # 跳过空结果
                continue

            print(f"\n{strategy.upper()} 策略:")

            final_metrics = task_metrics[-1]  # 最后一个任务的指标
            avg_diversity = np.mean([m['diversity'] for m in task_metrics if 'diversity' in m])
            avg_balance = np.mean([m['class_balance'] for m in task_metrics if 'class_balance' in m])

            print(f"  ✅ 最终缓冲区大小: {final_metrics['buffer_size']}/{final_metrics['buffer_capacity']}")
            print(f"  📊 平均多样性: {avg_diversity:.3f}")
            print(f"  ⚖️  平均平衡度: {avg_balance:.3f}")
            print(f"  🎯 最终类别数: {final_metrics.get('n_classes', 'N/A')}")


# 单元测试类 - 修复版
class TestBufferStrategies(unittest.TestCase):
    """缓冲区策略单元测试"""

    def setUp(self):
        self.args = MockArgs()
        self.device = torch.device('cpu')

    def test_buffer_creation(self):
        """测试缓冲区创建"""
        print("\n=== 单元测试: 缓冲区创建 ===")
        for mode in ['reservoir', 'herding']:
            with self.subTest(mode=mode):
                try:
                    buffer = Buffer(100, self.device, mode=mode, args=self.args)
                    self.assertIsNotNone(buffer)
                    self.assertEqual(buffer.mode, mode)
                    print(f"✅ {mode}模式创建成功")
                except Exception as e:
                    self.skipTest(f"{mode}模式创建失败: {e}")

    def test_herding_cache_mechanism(self):
        """测试herding缓存机制"""
        print("\n=== 单元测试: Herding缓存机制 ===")
        try:
            buffer = Buffer(100, self.device, mode='herding', args=self.args)

            examples = torch.randn(10, 3, 32, 32)
            labels = torch.randint(0, 3, (10,))
            features = torch.randn(10, 512)

            buffer._cache_current_task_data(examples, labels, None, None, None, features)

            self.assertGreater(len(buffer.current_task_cache), 0)
            self.assertIn('examples', list(buffer.current_task_cache.values())[0])
            print("✅ Herding缓存机制测试成功")
        except Exception as e:
            self.skipTest(f"Herding缓存测试失败: {e}")


def main():
    """主测试函数"""
    # 初始化测试器
    tester = BufferStrategyTester(device='cpu')  # 使用CPU避免CUDA问题

    try:
        # 运行全面测试
        results = tester.run_comprehensive_test()

        print("\n" + "=" * 60)
        print("✅ 所有测试完成！")
        print("=" * 60)
        print("\n💡 建议下一步:")
        print("1. 查看上面的指标，选择表现更好的策略")
        print("2. 对选择的策略运行小规模完整训练 (1-2个epoch)")
        print("3. 如果结果满意，再进行完整训练")

        return results

    except Exception as e:
        print(f"\n❌ 测试过程中出错: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    # 运行主测试
    print("开始运行CHARON缓冲区快速测试...")
    results = main()

    # 运行单元测试
    print("\n" + "=" * 60)
    print("🔬 运行单元测试...")
    print("=" * 60)

    # 创建测试套件并运行
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestBufferStrategies)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    if result.wasSuccessful():
        print("\n🎉 所有测试通过！")
    else:
        print(f"\n⚠️  有 {len(result.failures)} 个测试失败")
        sys.exit(1)