> 附录 B
> 
> 反射 API 的高级介绍
> 
> 反射允许代码在运行时检查类型、方法、字段、注解等，并将如何使用它们的决策从编译时推迟到运行时。为此，Java 的反射 API 提供了`Class`、`Field`、`Constructor`、`Method`、`Annotation`等类型。有了它们，就可以与编译时未知类型进行交互：例如，创建未知类的实例并在其上调用方法。
> 
> 反射及其用例可能会迅速变得复杂，我不会对其进行详细解释。相反，本附录旨在让你对反射是什么、在 Java 中看起来如何以及你可以或你的依赖项用它做什么有一个高级的理解。
> 
> 之后，你将准备好开始使用它或学习更长的教程，例如 Oracle 的《反射 API 教程》[`docs.oracle.com/javase/tutorial/reflect`](https://docs.oracle.com/javase/tutorial/reflect)。更重要的是，你将准备好理解模块系统对反射所做的更改，第 7.1.4 节和第十二章特别探讨了这一点。
> 
> 而不是从头开始构建，让我们从一个简单的例子开始。下面的代码片段创建了一个 URL，将其转换为字符串，然后打印出来。在求助于反射之前，我使用了普通的 Java 代码：
> 
> `URL url = new URL("http://codefx.org"); String urlString = url.toExternalForm(); System.out.println(urlString);`
> 
> 我在编译时（即，当我编写代码时）决定我想创建一个`URL`对象并在其中调用一个方法。尽管这不是最自然的方法，但你可以将前两行分成五个步骤：

1.  引用 URL 类。

1.  定位接受单个字符串参数的构造函数。

1.  用`http://codefx.org`调用它。

1.  定位`toExternalForm`方法。

1.  在`url`实例上调用它。

> 下面的列表展示了如何使用 Java 的反射 API 实现这五个步骤。
> 
> > 列表 B.1 使用反射创建`URL`并在其上调用`toExternalForm`
> > 
> `Class<?> urlClass = Class.forName("java.net.URL");` `①` `Constructor<?> urlConstructor = urlClass.getConstructor(String.class);` `②` `Object url = urlConstructor.newInstance("http://codefx.org");` `③` `Method toExternalFormMethod = urlClass.getMethod("toExternalForm");` `④` `Object methodCallResult = toExternalFormMethod.invoke(url);` `⑤`
> 
> > ①
> > 
> > 要操作类的 Class 实例是反射的入口。
> > 
> > ②
> > 
> > 获取接受 String 参数的构造函数
> > 
> > ③
> > 
> > 使用它以给定的字符串作为参数创建一个新的实例
> > 
> > ④
> > 
> > 获取`toExternalForm`方法
> > 
> > ⑤
> > 
> > 调用之前创建的实例中的方法
> > 
> 使用反射 API 比直接编写代码要繁琐。但这种方式，以前通常嵌入到代码中的细节（如使用 `URL` 或调用哪个方法）变成了字符串参数。因此，你不必在编译时决定使用 `URL` 和 `toExternalForm`，而可以在程序运行时决定选择哪种类型和方法。
> 
> 这种用法的大多数情况都发生在“框架化”环境中。以 JUnit 为例，它希望执行所有被 `@Test` 注解的方法。一旦找到它们，它就使用 `getMethod` 和 `invoke` 来调用它们。Spring 和其他 Web 框架在查找控制器和请求映射时也以类似的方式操作。希望在运行时加载用户提供的插件的可扩展应用程序是另一个用例。
> 
> 基本类型和方法
> 
> 反射 API 的入口是 `Class::forName`。在其简单形式中，这个静态方法接受一个完全限定的类名，并返回一个对应的 `Class` 实例。你可以使用这个实例来获取字段、方法、构造函数等。
> 
> 要获取特定的构造函数，使用构造函数参数的类型调用 `getConstructor` 方法，就像我之前做的那样。同样，可以通过调用 `getMethod` 并传递其名称以及参数类型来访问特定的方法。
> 
> 调用 `getMethod("toExternalForm")` 没有指定任何类型，因为该方法没有参数。这是 `URL.openConnection(Proxy)`，它接受一个 `Proxy` 参数：
> 
> `Class<?> urlClass = Class.forName("java.net.URL"); Method openConnectionMethod = urlClass.getMethod("openConnection", Proxy.class);`
> 
> `getConstructor` 和 `getMethod` 调用返回的实例分别是 `Constructor` 和 `Method` 类型。要调用底层成员，它们提供了如 `Constructor::newInstance` 和 `Method::invoke` 这样的方法。后者一个有趣的细节是，你需要将方法要调用的实例作为第一个参数传递。其他参数将被传递给被调用的方法。
> 
> 继续使用 `openConnection` 示例：
> 
> `openConnectionMethod.invoke(url, someProxy);`
> 
> 如果你想调用一个静态方法，实例参数将被忽略，可以是 `null`。
> 
> 除了 `Class`、`Constructor` 和 `Method` 之外，还有一个 `Field`，它允许对实例字段进行读写访问。使用实例调用 `get` 方法可以检索该字段在实例中的值——`set` 方法在指定的实例中设置指定的值。
> 
> `URL` 类有一个实例字段 `protocol`，其类型为 `String`；对于 URL [`codefx.org`](http://codefx.org)，它将包含 `"http"`。因为它私有，所以像这样的代码无法编译：
> 
> `URL url = new URL("http://codefx.org"); // 无法访问私有字段 ~> 编译错误 url.protocol = "https";`
> 
> 这是使用反射来完成相同任务的方法：
> 
> ``// `Class<?> urlClass` 和 `Object url` 与之前相同 Field protocolField = urlClass.getDeclaredField("protocol"); Object oldProtocol = protocolField.get(url); protocolField.set(url, "https");``
> 
> 虽然这可以编译，但它仍然会在 `get` 调用中导致 `IllegalAccessException`，因为 `protocol` 字段是私有的。但这并不意味着你不能继续。
> 
> 使用 `setAccessible` 突破 API
> 
> 反射的一个重要用例一直是通过访问非公共类型、方法和字段来突破 API。这被称为深度反射。开发者使用它来访问 API 不提供访问权限的数据，通过调整内部状态来解决依赖项中的错误，以及动态填充实例以正确的值——例如，Hibernate 就这样做。
> 
> 对于深度反射，你需要在使用之前对 `Method`、`Constructor` 或 `Field` 实例调用 `setAccessible(true)`：
> 
> ``// `Class<?> urlClass` 和 `Object url` 与之前相同 Field protocolField = urlClass.getDeclaredField("protocol"); protocolField.setAccessible(true); Object oldProtocol = field.get(url); protocolField.set(instance, "https");``
> 
> 当迁移到模块系统时，一个挑战是它剥夺了反射的超级能力，这意味着对 `setAccessible` 的调用更有可能失败。关于这一点以及如何补救，请参阅第十二章。
> 
> 注解标记代码以供反射
> 
> 注解是反射的重要组成部分。实际上，注解是为反射设计的。它们的目的是提供在运行时可以访问并在之后用于塑造程序行为的元信息。JUnit 的 `@Test` 和 Spring 的 `@Controller` 以及 `@RequestMapping` 是主要的例子。
> 
> 所有重要的反射相关类型，如 `Class`、`Field`、`Constructor`、`Method` 和 `Parameter` 都实现了 `AnnotatedElement` 接口。它的 Javadoc 包含了关于注解如何与这些元素相关联的详细解释（直接存在、间接存在或关联），但它的最简单形式是这样的：`getAnnotations` 方法返回一个 `Annotation` 实例数组，该数组的成员可以被访问。
> 
> 但在模块系统的背景下，你或你依赖的框架如何处理注解，这比它们仅通过反射来工作的基本事实要次要。这意味着任何带有注解的类在某个时刻都会被反射——如果这个类在模块中，这不一定能直接工作。
